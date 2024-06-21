#!/usr/bin/env python3

"""
Given BPE codes/vocab, create a lexicon (mapping of words to all possible BPE seqs).
"""

from __future__ import annotations

from argparse import ArgumentParser
from xml.etree import ElementTree

import _setup_returnn_env  # noqa
from returnn.util import bpe as bpe_utils


def parse_vocab(filename):
    """
    Can be either pure text file, line-based, or lexicon XML file, or Python vocab dict.

    :param str filename:
    :rtype: list[str]
    """
    if filename.endswith(".gz"):
        import gzip

        raw = gzip.open(filename, "r").read().decode("utf8")
    else:
        raw = open(filename, "r").read()
    if raw.startswith("{"):  # Python dict (str sym -> int idx)
        py_vocab = eval(raw)
        assert isinstance(py_vocab, dict)  # sym -> idx
        labels = {idx: label for (label, idx) in sorted(py_vocab.items())}
        min_label, max_label, num_labels = min(labels), max(labels), len(labels)
        assert 0 == min_label
        if num_labels - 1 < max_label:
            print("Vocab error: not all indices used? max label: %i" % max_label)
            print("unused labels: %r" % ([i for i in range(max_label + 1) if i not in labels],))
        assert num_labels - 1 == max_label
        zero_sym = labels[0]
        assert isinstance(zero_sym, str)
        return [label for (idx, label) in sorted(labels.items())]
    if raw.startswith("<?xml"):  # lexicon XML
        labels = []
        from io import StringIO

        raw_stream = StringIO(raw)
        context = iter(ElementTree.iterparse(raw_stream, events=("start", "end")))
        _, root = next(context)  # get root element
        for event, elem in context:
            if event == "end" and elem.tag == "lemma":
                for orth_elem in elem.findall("orth"):
                    orth = (orth_elem.text or "").strip()
                    labels.append(orth)
                root.clear()  # free memory
        return labels
    # Assume line-based. No idea how to to a good sanity check...
    return raw.splitlines()


def xml_prettify(element, indent="  "):
    """
    https://stackoverflow.com/a/38574067/133374 (deleted StackOverflow answer)

    :param ElementTree.Element element:
    :param str indent:
    """
    queue = [(0, element)]  # (level, element)
    while queue:
        level, element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = "\n" + indent * (level + 1)  # for child open
        if queue:
            element.tail = "\n" + indent * queue[0][0]  # for sibling open
        else:
            element.tail = "\n" + indent * (level - 1)  # for parent close
        queue[0:0] = children  # prepend so children come before siblings


def main():
    """
    Main entry.
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--bpe_vocab", required=True)
    arg_parser.add_argument("--word_vocab", required=True)
    arg_parser.add_argument("--unk")
    arg_parser.add_argument("--skip_special", action="store_true")
    arg_parser.add_argument("--lower_case", action="store_true")
    arg_parser.add_argument("--output")
    args = arg_parser.parse_args()

    bpe_syms = parse_vocab(args.bpe_vocab)
    words = parse_vocab(args.word_vocab)
    print("BPE symbols: num %i, first %r" % (len(bpe_syms), bpe_syms[0]))
    print("Words: num %i, first %r" % (len(words), words[0]))

    print("Build BPE prefix tree...")
    bpe = bpe_utils.PrefixTree(opts=bpe_utils.BpeOpts(label_postfix_merge_symbol=bpe_utils.BpePostMergeSymbol))
    for bpe_sym in bpe_syms:
        bpe.add(bpe_sym)

    print("Build lexicon...")
    xml = ElementTree.Element("lexicon")
    xml_phone_inventory = ElementTree.SubElement(xml, "phoneme-inventory")
    for bpe_sym in bpe_syms:  # each BPE symbol will be a phoneme in the XML
        xml_phone = ElementTree.SubElement(xml_phone_inventory, "phoneme")
        ElementTree.SubElement(xml_phone, "symbol").text = bpe_sym
        ElementTree.SubElement(xml_phone, "variation").text = "context"

    visited_words = set()

    # noinspection PyShadowingNames
    def visit_word(word):
        """
        :param str word:
        """
        if word in visited_words:
            return
        visited_words.add(word)
        bpe_sym_seqs = bpe_utils.CharSyncSearch(bpe=bpe, word=word).search()
        if not bpe_sym_seqs:
            print("no BPE seq found for word %r" % word)
            return
        xml_lemma = ElementTree.SubElement(xml, "lemma")
        ElementTree.SubElement(xml_lemma, "orth").text = word
        for bpe_sym_seq in bpe_sym_seqs:
            ElementTree.SubElement(xml_lemma, "phon").text = " ".join(bpe_sym_seq)

    for word in words:
        if args.lower_case:
            word = word.lower()
        if not word:
            continue
        if args.skip_special:
            if word.startswith("[") and word.endswith("]"):
                continue
            if word.startswith("<") and word.endswith(">"):
                continue
        visit_word(word)

    for bpe_sym in bpe_syms:
        if bpe_sym.endswith(bpe_utils.BpePostMergeSymbol):
            continue
        if bpe_sym not in words:
            continue
        # E.g. special symbols, which were skipped above.
        visit_word(bpe_sym)

    if args.output:
        xml_prettify(xml)
        xml_str = ElementTree.tostring(xml, encoding="utf-8")
        with open(args.output, "wb") as f:
            f.write(xml_str)
        print("Wrote XML:", args.output)
    else:
        print("Specify --output to save the XML.")


if __name__ == "__main__":
    from returnn.util import better_exchook

    better_exchook.install()
    main()
