#!/usr/bin/env python3

"""
Dump translation dataset to HDF.
"""

from __future__ import annotations

import sys
import argparse
import gzip
import pickle
import typing

import numpy
import h5py
from operator import itemgetter
from itertools import islice, zip_longest

UNKNOWN_LABEL = "<UNK>"
POSTFIX = "</S>"


class HDFTranslationDatasetCreator(object):
    """
    Creates the same HDF file as hdf_dump.py called on a TranslationDataset, but does so much faster
    and using much less memory.

    Input parameters are basically the files in a TranslationDataset folder. Additionally, the number of lines has to
    be given.
    """

    def __init__(
        self,
        hdf_file,
        source_file,
        target_file,
        source_vocabularies,
        target_vocabularies,
        source_factors,
        target_factors,
        number_of_lines,
        factor_separator="|",
        compression_method=None,
        line_buffer_size=100000,
        data_buffer_size=5000000,
    ):
        """
        :param str hdf_file: filename for the hdf file being created
        :param str source_file: filename of the source text file
        :param str target_file: filename of the target text file
        :param list[str] source_vocabularies: Filenames of the source vocabularies (in pickle format).
            Usually there is just one. In case of source factors, provide one per factor.
        :param list[str] target_vocabularies: Filenames of the target vocabularies (in pickle format).
            Usually there is just one. In case of target factors, provide one per factor.
        :param list[str] source_factors: Data keys for the source factors. First data key is always 'data'
            and must not be contained in the list.
        :param list[str] target_factors: Data keys for the target factors. First data key is always 'classes'
            and must not be contained in the list.
        :param str factor_separator: "String used to separate factors of the words.
            E.g. if "|", words are expected to be of format "<lemma>|<factor>|...".
        :param int number_of_lines: line count in source_file and target_file
        :param str compression_method: Optional compression method as supported by h5py.File.create_dataset().
            Applied to the main data only ('inputs' and 'target/data/classes').
        :param int line_buffer_size: number of corpus lines to read and process at once
        :param int data_buffer_size: space to reserve inside the hdf file at once, in numbers of integers
        """

        self.hdf_file_name = hdf_file
        self.hdf_file = None
        self.compression_method = compression_method

        self.source_file_handle = self._open_file(source_file)
        self.target_file_handle = self._open_file(target_file)

        source_vocabularies = [self._read_vocabulary(v) for v in source_vocabularies]
        target_vocabularies = [self._read_vocabulary(v) for v in target_vocabularies]

        self.source_data_keys = ["data"] + source_factors
        self.target_data_keys = ["classes"] + target_factors

        self._vocabularies = {"source": source_vocabularies, "target": target_vocabularies}
        self._vocabulary_sizes = {
            "source": [len(v) for v in source_vocabularies],
            "target": [len(v) for v in target_vocabularies],
        }
        self._unknown_ids = {
            "source": [v.get(UNKNOWN_LABEL) for v in source_vocabularies],
            "target": [v.get(UNKNOWN_LABEL) for v in target_vocabularies],
        }

        self.number_of_lines = number_of_lines
        self.line_buffer_size = line_buffer_size
        self.data_buffer_size = data_buffer_size

        self._number_of_processed_lines = 0
        self._write_offsets = {data_key: 0 for data_key in self.source_data_keys + self.target_data_keys}

        self.factor_separator = factor_separator

    def create(self):
        """
        Main function writing the HDF file.
        """
        self._init_hdf_file()

        print("Setting attributes...", file=sys.stderr)
        sys.stderr.flush()
        self._write_attributes()
        print("Done.", file=sys.stderr)

        print("Writing labels (vocabulary)...", file=sys.stderr)
        sys.stderr.flush()
        self._write_labels()
        print("Done.", file=sys.stderr)

        print("Writing source, target, sequence lengths and tags for all lines...", file=sys.stderr)
        sys.stderr.flush()
        end_of_file = False
        while not end_of_file:
            end_of_file = self._write_data()
            if not end_of_file:
                print("> Processed {} lines.".format(self._number_of_processed_lines), file=sys.stderr)
                sys.stderr.flush()
        print("Done.", file=sys.stderr)
        sys.stderr.flush()

        self.hdf_file.close()

    def _init_hdf_file(self):
        """
        Sets up the HDF file and initializes the datasets that will be filled.
        """
        self.hdf_file = h5py.File(self.hdf_file_name, "w")

        self.hdf_file.create_group("targets/data")
        self.hdf_file.create_group("targets/size")
        self.hdf_file.create_group("targets/labels")

        num_data_keys = len(self.source_data_keys) + len(self.target_data_keys)
        self.hdf_file.create_dataset("seqLengths", (self.number_of_lines, num_data_keys), dtype="int32")
        max_tag_length = len("line-") + len(str(self.number_of_lines))
        self._tag_dtype = "S{}".format(max_tag_length)
        self.hdf_file.create_dataset("seqTags", (self.number_of_lines,), dtype=self._tag_dtype)

        self.hdf_file.create_dataset(
            "inputs", (self.data_buffer_size,), maxshape=(None,), dtype="int32", compression=self.compression_method
        )

        # HDF format expects one input only, so store source factors as target too.
        for data_key in self.source_data_keys[1:] + self.target_data_keys:
            self.hdf_file["targets/data"].create_dataset(
                data_key, (self.data_buffer_size,), maxshape=(None,), dtype="int32", compression=self.compression_method
            )

    def _write_attributes(self):
        """
        Writes several attributes to the HDF file.
        """
        for index, data_key in enumerate(self.source_data_keys[1:], start=1):
            self.hdf_file["targets/size"].attrs[data_key] = (self._vocabulary_sizes["source"][index], 1)
        for index, data_key in enumerate(self.target_data_keys):
            self.hdf_file["targets/size"].attrs[data_key] = (self._vocabulary_sizes["target"][index], 1)

        # Those should be deprecated, but include nevertheless to exactly reproduce hdf_dump.
        self.hdf_file.attrs["inputPattSize"] = self._vocabulary_sizes["source"][0]
        self.hdf_file.attrs["numLabels"] = self._vocabulary_sizes["target"][0]

    def _write_labels(self):
        """
        Writes the labels (i.e. target vocabulary) to the HDF file.
        """
        for side in ["source", "target"]:
            # We have to write it for the source factors too, because they are treated like targets.

            data_keys = self.source_data_keys if side == "source" else self.target_data_keys
            for index, data_key in enumerate(data_keys):
                if data_key == "data":
                    continue

                sorted_vocabulary_tuples = sorted(self._vocabularies[side][index].items(), key=itemgetter(1))
                labels = [word.encode("utf8") for (word, _) in sorted_vocabulary_tuples]

                assert len(labels) == self._vocabulary_sizes[side][index], "Word ids were not unique."

                max_label_length = max([len(label) for label in labels])
                dtype = "S{}".format(max_label_length + 1)

                labels = [numpy.array(label, dtype=dtype, ndmin=1) for label in labels]
                labels = numpy.concatenate(labels)

                self.hdf_file["targets/labels"].create_dataset(
                    data_key, (self._vocabulary_sizes[side][index],), data=labels, dtype=dtype
                )

    def _write_data(self):
        """
        Loads a chunk of lines from the corpus and writes all corresponding data to the HDF file.

        :return: whether the end of the corpus is reached
        :rtype: bool
        """
        data_chunks = self._get_chunk()

        if not data_chunks["data"]:
            self._finalize_data()
            return True

        source_lengths = [len(line) for line in data_chunks["data"]]
        target_lengths = [len(line) for line in data_chunks["classes"]]

        chunk_size = len(data_chunks["data"])

        self._write_lengths(source_lengths, target_lengths)
        self._write_tags(chunk_size)

        for data_key, data_chunk in data_chunks.items():
            self._write_data_indices(data_chunk, data_key)

        self._number_of_processed_lines += chunk_size

        return False

    def _get_chunk(self):
        """
        Reads in the next chunk of lines from the corpus files.

        :return: a dict 'data_key' -> word indices for 'data_key' (int32, shape [Lines]) for all source and target data
        :rtype: dict[str,list[numpy.ndarray]]
        """
        source_lines = islice(self.source_file_handle, self.line_buffer_size)
        target_lines = islice(self.target_file_handle, self.line_buffer_size)

        data_chunks = {data_key: [] for data_key in self.source_data_keys + self.target_data_keys}
        for source_line, target_line in zip_longest(source_lines, target_lines):
            assert source_line is not None and target_line is not None, "Number of source and target lines differ."

            source_indices = self._line_to_indices(source_line, "source")
            for data_key, indices in zip(self.source_data_keys, source_indices):
                data_chunks[data_key].append(indices)

            target_indices = self._line_to_indices(target_line, "target")
            for data_key, indices in zip(self.target_data_keys, target_indices):
                data_chunks[data_key].append(indices)

        return data_chunks

    def _write_lengths(self, source_lengths, target_lengths):
        """
        Writes the sequence lengths to the HDF file.

        :param list[int] source_lengths: lengths of all source lines in current chunk
        :param list[int] target_lengths: lengths of all target lines in current chunk
        """
        # We treat source factors as targets internally because HDF format does not support multiple inputs.

        # For each sequence, seqLengths is expected to contain the length of the input and the length of each of the
        # targets, in alphabetical order of the target data keys. As all source and all target factors share the
        # same lengths we just choose between the source and target lengths.
        target_sequence_lengths = {}
        for data_key in self.source_data_keys[1:]:
            target_sequence_lengths[data_key] = source_lengths
        for data_key in self.target_data_keys:
            target_sequence_lengths[data_key] = target_lengths

        # Now sort by key.
        key_lengths_tuples_sorted = sorted(
            target_sequence_lengths.items(), key=lambda x: x[0]
        )  # type: typing.List[typing.Tuple[str,typing.List[int]]]  # nopep8
        target_lengths_sorted = [key_length_tuple[1] for key_length_tuple in key_lengths_tuples_sorted]

        # Finally, add one time the source lengths for the input ("data") and convert to numpy.
        lengths = numpy.array([source_lengths] + target_lengths_sorted, dtype="int32").transpose()

        offset = self._number_of_processed_lines
        assert len(lengths) + offset <= self.number_of_lines, "More lines in the corpus files than specified."

        self.hdf_file["seqLengths"][offset : offset + len(lengths), :] = lengths

    def _write_tags(self, chunk_size):
        """
        Writes the sequence tags to the HDF file.

        :param int chunk_size: number of lines in the current chunk
        """
        offset = self._number_of_processed_lines

        tags = [numpy.array("line-" + str(offset + i), dtype=self._tag_dtype, ndmin=1) for i in range(chunk_size)]
        tags = numpy.concatenate(tags)

        self.hdf_file["seqTags"][offset : offset + chunk_size] = tags

    def _write_data_indices(self, chunk, data_key):
        """
        Writes the main data (word indices for the source or target corpus) to the HDF file.

        :param list[numpy.ndarray] chunk: word indices for all lines in the current chunk
        :param str data_key: "data", "classes" or a name of a factor
        """
        indices = numpy.concatenate(chunk)

        if data_key == "data":
            dataset = self.hdf_file["inputs/"]
        else:
            dataset = self.hdf_file["targets/data/{}".format(data_key)]

        offset = self._write_offsets[data_key]
        length = len(indices)
        if offset + length > len(dataset):
            buffer_size = max(self.data_buffer_size, length)
            dataset.resize((offset + buffer_size,))

        dataset[offset : offset + length] = indices

        self._write_offsets[data_key] += length

    def _finalize_data(self):
        """
        Called after all data is written. Checks number of lines and resizes datasets down to actual data size.
        """
        # Make sure the number of lines given by the user was correct.
        # Otherwise lengths and labels would have trailing zeros.
        assert (
            self.number_of_lines == self._number_of_processed_lines
        ), "Fewer lines ({}) in the corpus files " "than specified ({}).".format(
            self._number_of_processed_lines, self.number_of_lines
        )

        # Trim datasets to actually occupied length, i.e. remove unused reserved space.
        self.hdf_file["inputs"].resize((self._write_offsets["data"],))
        for data_key in self.source_data_keys[1:] + self.target_data_keys:
            self.hdf_file["targets/data/{}".format(data_key)].resize((self._write_offsets[data_key],))

    def _line_to_indices(self, line, side):
        """
        Converts a line of text to arrays of word indices.

        :param str line: input line
        :param str side: "source" or "target"
        :return: word indices (int32, shape [num_words]) for all source or target factors
        :rtype: list[numpy.ndarray]
        """
        data_keys = self.source_data_keys if side == "source" else self.target_data_keys

        words = line.strip().split()

        if len(data_keys) == 1:
            word_list_per_factor = [words + [POSTFIX]]
        else:
            if words:
                words_split_into_factors = [word.split(self.factor_separator) for word in words]
                assert all(
                    len(factors) == len(data_keys) for factors in words_split_into_factors
                ), "All words must have all factors. Expected: " + self.factor_separator.join(data_keys)
                word_list_per_factor = zip(*words_split_into_factors)
            else:
                word_list_per_factor = [[]] * len(data_keys)
            word_list_per_factor = [list(words) + [POSTFIX] for words in word_list_per_factor]

        indices_list = []
        for index, (data_key, words) in enumerate(zip(data_keys, word_list_per_factor)):
            indices = [self._vocabularies[side][index].get(word, self._unknown_ids[side][index]) for word in words]

            indices_numpy = numpy.array(indices, dtype=numpy.int32)
            indices_list.append(indices_numpy)

        return indices_list

    @staticmethod
    def _open_file(file_name):
        """
        :param str file_name: filename of a plain text file, possibly zipped
        :return: file handle
        :rtype: io.TextIOWrapper|gzip.GzipFile
        """
        if file_name.endswith(".gz"):
            return gzip.open(file_name, "rt")
        else:
            return open(file_name, "r")

    @staticmethod
    def _read_vocabulary(file_name):
        """
        :param str file_name: filename of the vocabulary (in pickle format)
        :return: mapping from words to indices
        :rtype: dict[str,int]
        """
        file_handle = open(file_name, "rb")
        vocabulary = pickle.load(file_handle)
        return vocabulary


def parse_args():
    """
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_corpus", required=True, help="Source corpus file, possibly zipped.")
    parser.add_argument("-t", "--target_corpus", required=True, help="Target corpus file, possibly zipped.")
    parser.add_argument(
        "-v",
        "--source_vocabulary",
        required=True,
        help="Source vocabulary in pickle format."
        "In case of source factors provide a comma separated list containing vocabularies for each factor.",
    )
    parser.add_argument(
        "-w",
        "--target_vocabulary",
        required=True,
        help="Target vocabulary in pickle format."
        "In case of target factors provide a comma separated list containing vocabularies for each factor.",
    )
    parser.add_argument("-o", "--hdf_file", required=True, help="Output HDF file name.")
    parser.add_argument("--source_factors", help="Comma separated list of data keys for the source factors.")
    parser.add_argument("--target_factors", help="Comma separated list of data keys for the target factors.")
    parser.add_argument(
        "--factor_separator",
        default="|",
        help="String used to separate factors of the words, E.g. if '|', words are expected to be "
        "of format '<lemma>|<factor>|...'",
    )
    parser.add_argument(
        "-n", "--number_of_lines", required=True, type=int, help="The number of total lines in the corpus files."
    )
    parser.add_argument(
        "-c", "--compression", help="Type of compression (e.g. 'gzip', 'lzf'). Turned off if not given."
    )
    parser.add_argument("-l", "--line_buffer_size", type=int, help="How many lines to read at once.", default=100000)
    parser.add_argument(
        "-d",
        "--data_buffer_size",
        type=int,
        help="How much space to reserve in the HDF dataset " "at once (in number of integers).",
        default=5000000,
    )

    return parser.parse_args()


def main():
    """
    Main entry.
    """
    args = parse_args()

    # In case of source or target factors we need a vocabularies for each.
    source_vocabularies = args.source_vocabulary.split(",")
    target_vocabularies = args.target_vocabulary.split(",")

    source_factors = args.source_factors.split(",") if args.source_factors else []
    target_factors = args.target_factors.split(",") if args.target_factors else []

    assert len(source_factors) + 1 == len(source_vocabularies), (
        "Number of source factors must be one less "
        "than number of source vocabularies (first factor is always called 'data')"
    )
    assert len(target_factors) + 1 == len(target_vocabularies), (
        "Number of target factors must be one less "
        "than number of target vocabularies (first factor is always called 'classes')"
    )

    HDFTranslationDatasetCreator(
        args.hdf_file,
        args.source_corpus,
        args.target_corpus,
        source_vocabularies,
        target_vocabularies,
        source_factors,
        target_factors,
        args.number_of_lines,
        args.factor_separator,
        args.compression,
        args.line_buffer_size,
        args.data_buffer_size,
    ).create()


if __name__ == "__main__":
    sys.exit(main())
