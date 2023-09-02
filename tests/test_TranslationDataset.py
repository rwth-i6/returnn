# coding: utf8
from __future__ import unicode_literals

import sys
import os
import unittest
import tempfile
import shutil
import gzip
import pickle

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

import _setup_test_env  # noqa
from returnn.util import better_exchook
from returnn.datasets.lm import TranslationDataset, TranslationFactorsDataset
from returnn.util.basic import init_thread_join_hack

better_exchook.install()
better_exchook.replace_traceback_format_tb()
init_thread_join_hack()


dummy_source_text = (
    "This is some example text.\n"
    "It is used to test the translation dataset.\n"
    "We will write it into a temporary file.\n"
)

dummy_target_text = (
    "Das ist ein Beispieltext.\n"
    "Er wird zum Testen des TranslationDatasets benutzt.\n"
    "Wir werden ihn in eine temporäre Datei schreiben.\n"
)


def create_vocabulary(text):
    """
    :param str text: any natural text
    :return: mapping of words in the text to ids, as well as the inverse mapping
    :rtype: (dict[str, int], dict[int, str])
    """
    vocabulary = {word: index for index, word in enumerate(set(text.strip().split()))}
    inverse_vocabulary = {index: word for word, index in vocabulary.items()}

    return vocabulary, inverse_vocabulary


def word_ids_to_sentence(word_ids, vocabulary):
    """
    :param list[int] word_ids:
    :param dict[int, str] vocabulary: mapping from word ids to words
    :return: concatenation of all words
    :rtype: str
    """
    words = [vocabulary[word_id] for word_id in word_ids]
    return " ".join(words)


def test_translation_dataset():
    """
    Checks whether a dummy translation dataset can be read and whether the returned word indices are correct.
    We create the necessary corpus and vocabulary files on the fly.
    """

    dummy_dataset = tempfile.mkdtemp()
    source_file_name = os.path.join(dummy_dataset, "source.test.gz")  # testing both zipped and unzipped
    target_file_name = os.path.join(dummy_dataset, "target.test")

    with gzip.open(source_file_name, "wb") as source_file:  # writing in binary format works both in Python 2 and 3
        source_file.write(dummy_source_text.encode("utf-8"))

    with open(target_file_name, "wb") as target_file:
        target_file.write(dummy_target_text.encode("utf-8"))

    for postfix in ["", " </S>"]:  # test with and without postfix

        # Replace one word by <UNK>.
        # This way it will not appear in the vocabulary (and <UNK> is added to the vocabulary).
        # We will test below whether this word is assigned the unknown id by checking whether the reconstruction also
        # contains <UNK>. Note, that the input file is already written and contains the original word.
        dummy_target_text_with_unk = dummy_target_text.replace("TranslationDatasets", "<UNK>")

        # Append postfix just to have it in the vocabulary
        source_vocabulary, inverse_source_vocabulary = create_vocabulary(dummy_source_text + postfix)
        target_vocabulary, inverse_target_vocabulary = create_vocabulary(dummy_target_text_with_unk + postfix)

        source_vocabulary_file_name = os.path.join(dummy_dataset, "source.vocab.pkl")
        target_vocabulary_file_name = os.path.join(dummy_dataset, "target.vocab.pkl")

        with open(source_vocabulary_file_name, "wb") as source_vocabulary_file:
            pickle.dump(source_vocabulary, source_vocabulary_file)

        with open(target_vocabulary_file_name, "wb") as target_vocabulary_file:
            pickle.dump(target_vocabulary, target_vocabulary_file)

        translation_dataset = TranslationDataset(
            path=dummy_dataset,
            file_postfix="test",
            source_postfix=postfix,
            target_postfix=postfix,
            unknown_label={"classes": "<UNK>"},
        )
        translation_dataset.init_seq_order(epoch=1)
        translation_dataset.load_seqs(0, 10)

        num_seqs = len(dummy_source_text.splitlines())
        assert_equal(translation_dataset.num_seqs, num_seqs)

        # Reconstruct the sentences from the word ids and compare with input.
        for sequence_index in range(num_seqs):
            source_word_ids = translation_dataset.get_data(sequence_index, "data")
            source_sentence = word_ids_to_sentence(source_word_ids, inverse_source_vocabulary)
            assert_equal(source_sentence, dummy_source_text.splitlines()[sequence_index] + postfix)

            target_word_ids = translation_dataset.get_data(sequence_index, "classes")
            target_sentence = word_ids_to_sentence(target_word_ids, inverse_target_vocabulary)
            assert_equal(target_sentence, dummy_target_text_with_unk.splitlines()[sequence_index] + postfix)

    shutil.rmtree(dummy_dataset)


num_source_factors = 2
dummy_source_text_factor_0 = "This is some example text.\n" "The factors here have no meaning\n"
dummy_source_text_factor_1 = "a b c d e\n" "a b c d e f\n"
dummy_source_text_factored_format = (
    "This|a is|b some|c example|d text.|e\n" "The|a factors|b here|c have|d no|e meaning|f\n"
)

num_target_factors = 3
dummy_target_text_factor_0 = "Das ist ein Beispieltext.\n" "Die Factors hier haben keinen Sinn.\n"
dummy_target_text_factor_1 = "a b c d\n" "a b c d e f\n"
dummy_target_text_factor_2 = "1 2 3 4\n" "1 2 3 4 5 6\n"
dummy_target_text_factored_format = (
    "Das|a|1 ist|b|2 ein|c|3 Beispieltext.|d|4\n" "Die|a|1 Factors|b|2 hier|c|3 haben|d|4 keinen|e|5 Sinn.|f|6\n"
)


def test_translation_factors_dataset():
    """
    Similar to test_translation_dataset(), but using translation factors.
    """
    source_text_per_factor = [dummy_source_text_factor_0, dummy_source_text_factor_1]
    target_text_per_factor = [dummy_target_text_factor_0, dummy_target_text_factor_1, dummy_target_text_factor_2]

    source_vocabulary_names = ["source.vocab.pkl", "source_factor1.vocab.pkl"]
    target_vocabulary_names = ["target.vocab.pkl", "target_factor1.vocab.pkl", "target_factor2.vocab.pkl"]

    source_data_keys = ["data", "source_factor1"]
    target_data_keys = ["classes", "target_factor1", "target_factor2"]

    dummy_dataset = tempfile.mkdtemp()
    source_file_name = os.path.join(dummy_dataset, "source.test.gz")  # testing both zipped and unzipped
    target_file_name = os.path.join(dummy_dataset, "target.test")

    with gzip.open(source_file_name, "wb") as source_file:  # writing in binary format works both in Python 2 and 3
        source_file.write(dummy_source_text_factored_format.encode("utf8"))

    with open(target_file_name, "wb") as target_file:
        target_file.write(dummy_target_text_factored_format.encode("utf8"))

    for postfix in ["", " </S>"]:  # test with and without postfix
        vocabularies, inverse_vocabularies = [], []
        for dummy_text in source_text_per_factor + target_text_per_factor:
            # Append postfix just to have it in the vocabulary
            vocabulary, inverse_vocabulary = create_vocabulary(dummy_text + postfix)
            vocabularies.append(vocabulary)
            inverse_vocabularies.append(inverse_vocabulary)

        vocabulary_names = source_vocabulary_names + target_vocabulary_names
        for index, vocabulary in enumerate(vocabularies):
            with open(os.path.join(dummy_dataset, vocabulary_names[index]), "wb") as vocabulary_file:
                pickle.dump(vocabulary, vocabulary_file)

        translation_dataset = TranslationFactorsDataset(
            path=dummy_dataset,
            file_postfix="test",
            factor_separator="|",
            source_factors=source_data_keys[1:],
            target_factors=target_data_keys[1:],
            source_postfix=postfix,
            target_postfix=postfix,
        )

        translation_dataset.init_seq_order(epoch=1)
        translation_dataset.load_seqs(0, 10)

        num_seqs = len(dummy_target_text_factored_format.splitlines())
        assert_equal(translation_dataset.num_seqs, num_seqs)

        # Reconstruct the sentences from the word ids for all factors and compare with input.
        data_keys = source_data_keys + target_data_keys
        texts_per_factor = source_text_per_factor + target_text_per_factor
        for index, text in enumerate(texts_per_factor):
            for sequence_index in range(num_seqs):
                word_ids = translation_dataset.get_data(sequence_index, data_keys[index])
                sentence = word_ids_to_sentence(word_ids, inverse_vocabularies[index])
                assert_equal(sentence, text.splitlines()[sequence_index] + postfix)

    shutil.rmtree(dummy_dataset)


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
