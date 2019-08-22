#!/usr/bin/env python3

import sys
import argparse
import gzip
import pickle

import numpy
import h5py
from operator import itemgetter
from itertools import islice, zip_longest

UNKNOWN_LABEL = "<UNK>"
POSTFIX = " </S>"


class HDFTranslationDatasetCreator(object):
  """
  Creates the same HDF file as hdf_dump.py called on a TranslationDataset, but does so much faster
  and using much less memory.

  Input parameters are basically the files in a TranslationDataset folder. Additionally, the number of lines has to
  be given.
  """

  def __init__(self, hdf_file, source_file, target_file, source_vocabulary, target_vocabulary, number_of_lines,
               compression_method=None, line_buffer_size=100000, data_buffer_size=5000000):
    """
    :param str hdf_file: filename for the hdf file being created
    :param str source_file: filename of the source text file
    :param str target_file: filename of the target text file
    :param str source_vocabulary: filename of the source vocabulary (in pickle format)
    :param str target_vocabulary: filename of the target vocabulary (in pickle format)
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

    source_vocabulary = self._read_vocabulary(source_vocabulary)
    target_vocabulary = self._read_vocabulary(target_vocabulary)

    self._vocabularies = {"source": source_vocabulary, "target": target_vocabulary}
    self._vocabulary_sizes = {"source": len(source_vocabulary), "target": len(target_vocabulary)}
    self._unknown_ids = {"source": source_vocabulary[UNKNOWN_LABEL], "target": target_vocabulary[UNKNOWN_LABEL]}

    self.number_of_lines = number_of_lines
    self.line_buffer_size = line_buffer_size
    self.data_buffer_size = data_buffer_size

    self._number_of_processed_lines = 0
    self._write_offsets = {"source": 0, "target": 0}

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

    self.hdf_file.create_dataset("seqLengths", (self.number_of_lines, 2), dtype="int32")
    max_tag_length = len("line-") + len(str(self.number_of_lines))
    self._tag_dtype = "S{}".format(max_tag_length)
    self.hdf_file.create_dataset("seqTags", (self.number_of_lines,), dtype=self._tag_dtype)

    self.hdf_file.create_dataset("inputs", (self.data_buffer_size,), maxshape=(None,),
                                 dtype="int32", compression=self.compression_method)
    self.hdf_file["targets/data"].create_dataset("classes", (self.data_buffer_size,), maxshape=(None,),
                                                 dtype="int32", compression=self.compression_method)

  def _write_attributes(self):
    """
    Writes several attributes to the HDF file.
    """
    self.hdf_file["targets/size"].attrs["classes"] = (self._vocabulary_sizes["target"], 1)

    # Those should be deprecated, but include nevertheless to exactly reproduce hdf_dump.
    self.hdf_file.attrs["inputPattSize"] = self._vocabulary_sizes["source"]
    self.hdf_file.attrs["numLabels"] = self._vocabulary_sizes["target"]

  def _write_labels(self):
    """
    Writes the labels (i.e. target vocabulary) to the HDF file.
    """
    sorted_vocabulary_tuples = sorted(self._vocabularies["target"].items(), key=itemgetter(1))
    labels = [word.encode("utf8") for (word, _) in sorted_vocabulary_tuples]

    assert len(labels) == self._vocabulary_sizes["target"], "Word ids were not unique."

    max_label_length = max([len(label) for label in labels])
    dtype = "S{}".format(max_label_length + 1)

    labels = [numpy.array(label, dtype=dtype, ndmin=1) for label in labels]
    labels = numpy.concatenate(labels)

    self.hdf_file["targets/labels"].create_dataset("classes", (self._vocabulary_sizes["target"],),
                                                   data=labels, dtype=dtype)

  def _write_data(self):
    """
    Loads a chunk of lines from the corpus and writes all corresponding data to the HDF file.

    :return: whether the end of the corpus is reached
    :rtype: bool
    """
    source_chunk, target_chunk = self._get_chunk()

    if not source_chunk:
      self._finalize_data()
      return True

    source_lengths = [len(line) for line in source_chunk]
    target_lengths = [len(line) for line in target_chunk]

    chunk_size = len(source_chunk)

    self._write_lengths(source_lengths, target_lengths)
    self._write_tags(chunk_size)

    self._write_data_indices(source_chunk, "source")
    self._write_data_indices(target_chunk, "target")

    self._number_of_processed_lines += chunk_size

    return False

  def _get_chunk(self):
    """
    Reads in the next chunk of lines from the corpus files.

    :return: word indices for source and target, int32, shape [Lines]
    :rtype: list[numpy.ndarray], list[numpy.ndarray]
    """
    source_lines = islice(self.source_file_handle, self.line_buffer_size)
    target_lines = islice(self.target_file_handle, self.line_buffer_size)

    source_chunk = []
    target_chunk = []
    for source_line, target_line in zip_longest(source_lines, target_lines):
      assert source_line is not None and target_line is not None, "Number of source and target lines differ."

      source_indices = self._line_to_indices(source_line, "source")
      target_indices = self._line_to_indices(target_line, "target")

      source_chunk.append(source_indices)
      target_chunk.append(target_indices)

    return source_chunk, target_chunk

  def _write_lengths(self, source_lengths, target_lengths):
    """
    Writes the sequence lengths to the HDF file.

    :param list[int] source_lengths: lengths of all source lines in current chunk
    :param list[int] target_lengths: lengths of all target lines in current chunk
    """
    lengths = numpy.array([source_lengths, target_lengths], dtype="int32").transpose()

    offset = self._number_of_processed_lines
    assert len(lengths) + offset <= self.number_of_lines, "More lines in the corpus files than specified."

    self.hdf_file["seqLengths"][offset:offset + len(lengths), :] = lengths

  def _write_tags(self, chunk_size):
    """
    Writes the sequence tags to the HDF file.

    :param int chunk_size: number of lines in the current chunk
    """
    offset = self._number_of_processed_lines

    tags = [numpy.array("line-" + str(offset + i), dtype=self._tag_dtype, ndmin=1) for i in range(chunk_size)]
    tags = numpy.concatenate(tags)

    self.hdf_file["seqTags"][offset: offset + chunk_size] = tags

  def _write_data_indices(self, chunk, side):
    """
    Writes the main data (word indices for the source or target corpus) to the HDF file.

    :param list[numpy.ndarray] chunk: word indices for all lines in the current chunk
    :param str side: "source" or "target"
    """
    indices = numpy.concatenate(chunk)

    if side == "source":
      dataset = self.hdf_file["inputs/"]
    else:
      dataset = self.hdf_file["targets/data/classes"]

    offset = self._write_offsets[side]
    length = len(indices)
    if offset + length > len(dataset):
      buffer_size = max(self.data_buffer_size, length)
      dataset.resize((offset + buffer_size,))

    dataset[offset:offset + length] = indices

    self._write_offsets[side] += length

  def _finalize_data(self):
    """
    Called after all data is written. Checks number of lines and resizes datasets down to actual data size.
    """
    # Make sure the number of lines given by the user was correct.
    # Otherwise lengths and labels would have trailing zeros.
    assert self.number_of_lines == self._number_of_processed_lines, "Fewer lines ({}) in the corpus files " \
        "than specified ({}).".format(self._number_of_processed_lines, self.number_of_lines)

    # Trim datasets to actually occupied length, i.e. remove unused reserved space.
    self.hdf_file["inputs"].resize((self._write_offsets["source"],))
    self.hdf_file["targets/data/classes"].resize((self._write_offsets["target"],))

  def _line_to_indices(self, line, side):
    """
    Converts a line of text to an array of word indices.

    :param str line: input line
    :param str side: "source" or "target"
    :return: word indices, int32, shape [num_words]
    :rtype: numpy.ndarray
    """
    line = line.strip() + POSTFIX
    words = line.split()

    indices = [self._vocabularies[side].get(word, self._unknown_ids[side]) for word in words]

    indices_numpy = numpy.array(indices, dtype=numpy.int32)
    return indices_numpy

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
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--source_corpus", required=True, help="Source corpus file, possibly zipped.")
  parser.add_argument("-t", "--target_corpus", required=True, help="Target corpus file, possibly zipped.")
  parser.add_argument("-v", "--source_vocabulary", required=True, help="Source vocabulary in pickle format.")
  parser.add_argument("-w", "--target_vocabulary", required=True, help="Target vocabulary in pickle format.")
  parser.add_argument("-o", "--hdf_file", required=True, help="Output HDF file name.")
  parser.add_argument("-n", "--number_of_lines", required=True, type=int,
                      help="The number of total lines in the corpus files.")
  parser.add_argument("-c", "--compression", help="Type of compression (e.g. 'gzip', 'lzf'). Turned off if not given.")
  parser.add_argument("-l", "--line_buffer_size", type=int, help="How many lines to read at once.", default=100000)
  parser.add_argument("-d", "--data_buffer_size", type=int, help="How much space to reserve in the HDF dataset "
                      "at once (in number of integers).", default=5000000)

  return parser.parse_args()


def main():
  args = parse_args()

  HDFTranslationDatasetCreator(args.hdf_file, args.source_corpus, args.target_corpus,
                               args.source_vocabulary, args.target_vocabulary, args.number_of_lines,
                               args.compression, args.line_buffer_size, args.data_buffer_size).create()


if __name__ == "__main__":
  sys.exit(main())
