"""
tests for HDF dataset
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import os
import sys
import _setup_test_env  # noqa

from returnn.datasets import Dataset
from returnn.datasets.hdf import *
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises
import returnn.util.basic as util
import h5py
import numpy as np
import os
import unittest
from returnn.util import better_exchook


my_dir = os.path.dirname(os.path.abspath(__file__))


def test_HDFDataset_init():
    """
    This method tests initialization of the HDFDataset class
    """
    toy_dataset = HDFDataset()
    assert_equal(toy_dataset.file_start, [0], "self.file_start init problem, should be [0]")
    assert_equal(toy_dataset.files, [], "self.files init problem, should be []")
    assert_equal(toy_dataset.file_seq_start, [], "self.file_seq_start init problem, should be []")
    return toy_dataset


def generate_dummy_hdf(num_datasets=1):
    filenames = []
    for idx in range(1, num_datasets + 1):
        fn = get_test_tmp_file(".%i.hdf5" % idx)
        filenames.append(fn)
        dataset = h5py.File(fn, "w")
        dataset.create_group("streams")

        dataset["streams"].create_group("features")
        dataset["streams"]["features"].attrs["parser"] = "feature_sequence"
        dataset["streams"]["features"].create_group("data")

        dataset["streams"].create_group("classes")
        dataset["streams"]["classes"].attrs["parser"] = "sparse"
        dataset["streams"]["classes"].create_group("data")

        import string

        random = np.random.RandomState(42)
        feature_size = 13
        seq_names = ["dataset_%d_sequence_%d" % (idx, i) for i in range(100)]
        class_names = list(string.ascii_lowercase)
        num_classes = len(class_names)
        print(class_names, len(class_names))
        for idx in range(100):
            class_id = random.randint(low=0, high=num_classes)
            seq_len = random.randint(low=1, high=20)
            features = random.rand(seq_len, feature_size)
            dataset["streams"]["features"]["data"].create_dataset(name=seq_names[idx], data=features, dtype="float32")
            dataset["streams"]["classes"]["data"].create_dataset(
                name=seq_names[idx], shape=(1,), data=np.int32(class_id), dtype="int32"
            )

        dt = h5py.special_dtype(vlen=str)
        feature_names = dataset["streams"]["classes"].create_dataset(
            "feature_names", shape=(len(class_names),), dtype=dt
        )
        for id_x, orth in enumerate(class_names):
            feature_names[id_x] = orth

        dt = h5py.special_dtype(vlen=str)
        sequence_names_data = dataset.create_dataset("seq_names", shape=(len(seq_names),), dtype=dt)
        for ind, val in enumerate(seq_names):
            sequence_names_data[ind] = val

        dataset.close()
    return filenames


# Note that nosetests might even call this function, as it has "test" in its name... Does not matter, though.
def get_test_tmp_file(suffix=".hdf"):
    """
    :param str suffix: e.g. ".hdf"
    :return: filename
    :rtype: str
    """
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    fn = f.name
    import atexit

    atexit.register(lambda: os.remove(fn))
    return fn


_hdf_cache = {}  # opts -> hdf fn


def generate_hdf_from_other(opts: Dict[str, Any], suffix: str = ".hdf", *, use_cache: bool = True) -> str:
    """
    :param opts:
    :param suffix:
    :param use_cache:
    :return: hdf filename
    """
    # See test_hdf_dump.py and tools/hdf_dump.py.
    from returnn.util.basic import make_hashable

    cache_key = None
    if use_cache:
        cache_key = make_hashable(opts)
        if cache_key in _hdf_cache:
            return _hdf_cache[cache_key]
    fn = get_test_tmp_file(suffix=suffix)
    from returnn.datasets.basic import init_dataset

    dataset = init_dataset(opts)
    hdf_dataset = HDFDatasetWriter(fn)
    hdf_dataset.dump_from_dataset(dataset)
    hdf_dataset.close()
    if use_cache:
        _hdf_cache[cache_key] = fn
    return fn


def generate_hdf_from_dummy():
    """
    :return: hdf filename
    :rtype: str
    """
    return generate_hdf_from_other(
        {"class": "DummyDataset", "input_dim": 13, "output_dim": 7, "num_seqs": 23, "seq_len": 17}
    )


def test_hdf_dump():
    generate_hdf_from_dummy()


class DatasetTestReader:
    def __init__(self, dataset):
        """
        :param Dataset dataset:
        """
        self.dataset = dataset
        self.data_keys = None
        self.data_shape = {}  # key -> shape
        self.data_sparse = {}  # key -> bool
        self.data_dtype = {}  # key -> str
        self.data = {}  # type: typing.Dict[str,typing.List[numpy.ndarray]]  # key -> list
        self.seq_lens = []  # type: typing.List[util.NumbersDict]
        self.seq_tags = []
        self.num_seqs = 0

    def read_all(self, epoch=1):
        """
        :param int epoch:
        """
        dataset = self.dataset
        dataset.init_seq_order(epoch=epoch)
        data_keys = dataset.get_data_keys()
        self.data_keys = data_keys
        self.data_shape = {key: dataset.get_data_shape(key) for key in data_keys}
        self.data_sparse = {key: dataset.is_data_sparse(key) for key in data_keys}
        self.data_dtype = {key: dataset.get_data_dtype(key) for key in data_keys}
        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            for key in data_keys:
                data = dataset.get_data(seq_idx=seq_idx, key=key)
                self.data.setdefault(key, []).append(data)
            seq_len = dataset.get_seq_length(seq_idx=seq_idx)
            self.seq_lens.append(seq_len)
            seq_tag = dataset.get_tag(seq_idx)
            self.seq_tags.append(seq_tag)
            seq_idx += 1
        print("Iterated through %r, num seqs %i" % (dataset, seq_idx))
        self.num_seqs += seq_idx


def test_hdf_dump_not_frame_synced():
    num_seqs = 3
    from returnn.datasets.generating import TaskNumberBaseConvertDataset

    hdf_fn = generate_hdf_from_other({"class": "TaskNumberBaseConvertDataset", "num_seqs": num_seqs})
    hdf = HDFDataset([hdf_fn])
    orig = TaskNumberBaseConvertDataset(num_seqs=num_seqs)
    hdf_reader = DatasetTestReader(hdf)
    orig_reader = DatasetTestReader(orig)
    hdf_reader.read_all()
    orig_reader.read_all()
    assert hdf_reader.data_keys == orig_reader.data_keys == ["data", "classes"]
    assert hdf_reader.num_seqs == orig_reader.num_seqs == num_seqs
    for seq_idx in range(num_seqs):
        # Not synced, i.e. different lengths:
        assert_not_equal(orig_reader.seq_lens[seq_idx]["data"], orig_reader.seq_lens[seq_idx]["classes"])
        for key in orig_reader.data_keys:
            assert_equal(hdf_reader.seq_lens[seq_idx][key], orig_reader.seq_lens[seq_idx][key])
            assert_equal(hdf_reader.data[key][seq_idx].tolist(), orig_reader.data[key][seq_idx].tolist())


def test_HDFDataset_partition_epoch():
    partition_epoch = 3
    num_seqs = 11
    from returnn.datasets.generating import TaskNumberBaseConvertDataset

    hdf_fn = generate_hdf_from_other({"class": "TaskNumberBaseConvertDataset", "num_seqs": num_seqs})
    hdf = HDFDataset([hdf_fn], partition_epoch=partition_epoch)
    orig = TaskNumberBaseConvertDataset(num_seqs=num_seqs)
    hdf_reader = DatasetTestReader(hdf)
    orig_reader = DatasetTestReader(orig)
    for epoch in range(1, partition_epoch + 1):
        hdf_reader.read_all(epoch=epoch)
    orig_reader.read_all()  # single epoch
    assert hdf_reader.data_keys == orig_reader.data_keys == ["data", "classes"]
    assert hdf_reader.num_seqs == orig_reader.num_seqs == num_seqs
    for seq_idx in range(num_seqs):
        # Not synced, i.e. different lengths:
        assert_not_equal(orig_reader.seq_lens[seq_idx]["data"], orig_reader.seq_lens[seq_idx]["classes"])
        for key in orig_reader.data_keys:
            assert_equal(hdf_reader.seq_lens[seq_idx][key], orig_reader.seq_lens[seq_idx][key])
            assert_equal(hdf_reader.data[key][seq_idx].tolist(), orig_reader.data[key][seq_idx].tolist())


def test_SimpleHDFWriter():
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    n_dim = 13
    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None)
    seq_lens1 = [11, 7, 5]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens1), max(seq_lens1), n_dim)).astype("float32"),
        seq_len=seq_lens1,
        seq_tag=["seq-%i" % i for i in range(len(seq_lens1))],
    )
    seq_lens2 = [10, 13, 3, 2]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens2), max(seq_lens2), n_dim)).astype("float32"),
        seq_len=seq_lens2,
        seq_tag=["seq-%i" % (i + len(seq_lens1)) for i in range(len(seq_lens2))],
    )
    writer.close()
    seq_lens = seq_lens1 + seq_lens2

    dataset = HDFDataset(files=[fn])
    assert dataset.get_data_keys() == ["data"]
    assert dataset.get_target_list() == []
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == [n_dim]
    assert reader.data_dtype["data"] == "float32"
    assert len(seq_lens) == reader.num_seqs
    for i, seq_len in enumerate(seq_lens):
        assert reader.seq_lens[i]["data"] == seq_len
    print("tags:", reader.seq_tags)
    assert_equal(reader.seq_tags, ["seq-%i" % i for i in range(reader.num_seqs)])
    assert isinstance(reader.seq_tags[0], str)


def test_SimpleHDFWriter_small():
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    n_dim = 3
    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None)
    seq_lens = [2, 3]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens), max(seq_lens), n_dim)).astype("float32"),
        seq_len=seq_lens,
        seq_tag=["seq-%i" % i for i in range(len(seq_lens))],
    )
    writer.close()

    dataset = HDFDataset(files=[fn])
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == [n_dim]
    assert reader.data_dtype["data"] == "float32"
    assert len(seq_lens) == reader.num_seqs
    for i, seq_len in enumerate(seq_lens):
        assert reader.seq_lens[i]["data"] == seq_len

    if sys.version_info[0] >= 3:  # gzip.compress is >=PY3
        print("raw content (gzipped):")
        import gzip

        print(repr(gzip.compress(open(fn, "rb").read())))


def test_read_simple_hdf():
    if sys.version_info[0] <= 2:  # gzip.decompress is >=PY3
        raise unittest.SkipTest
    # n_dim, seq_lens, raw_gzipped is via test_SimpleHDFWriter_small
    n_dim = 3
    seq_lens = [2, 3]
    raw_gzipped = (
        b"\x1f\x8b\x08\x00\x80\xc8f\\\x02\xff\xed\x9a=l\xd3@\x14\xc7\xefl'XQ\x8a\xd2vhTT\xf0\x84X\x90\xc2V1\xb8TJ\xa0C"
        b'\x81\x8a\x04\xa9\x03BM\x85i"5Q\x82]$`h%\x18@'
        b"\xea\xc6\x02\x1b\x03\x03\x1f\x0b\x0b\x1bm\x10S\xa7NLL\xb01\x96\x1d"
        b"\xa9\xd8\xbewI}\x8d\x9d\xb1i\xf9\xff\x86\\\xee\xf2\x9e\xef\xc3\xff\xbb{"
        b"\xf6\xe5\xc5\\\xf1\xeaHf2\xc3\x02L\x93\x19,"
        b"\xc7\x0e\xb2O|\xb7\xa3y\xf9\xfb\x12\xa5\x9c"
        b"\xd2\xe7\x94\xbe\xd3d\xb9\x19\xfe\x96\xa7\xf2\x1c]\xdf\xd2E\xbeF\x8e"
        b"\x95[\xa5R`\xbd\xaf \xeby\x95\x12\xe9\x05\x06\xfeG\xe6J\xb3\x0bA\xbaH\xf9\x02\xa5;Z\xd4n\xb5\xba\xec\xac\xba"
        b"\x8c\xb9N{\xdei\xaex5W\x94\xd7\x9b\xad5O\x94W\xaa+nW\xaf\x83\xf44JzUu\x9de\xd3\xfe\\\t\x14{"
        b"\xda\xffn\x8a\xeb/T=\xaf\\\x7f\xec\x04:7\xfd\xe9\x14Z^\x89\xcc\x0f\x92\xbd\xefS "
        b"\x7f3\xf4o\xae5\x8a\xf5\x86\x1b\xeb\xc7\x99ZoF\xfa\xcdS"
        b"\x97\xc5\xfc\x1aX\xaf\xf4\x1f\x91\xfe\x95z\xc3q=\xa7\xe5"
        b"&\xf9\xa7\xe2\xdb]v\xda\xf1\xed\xee\xdd\x9e|\xe28s\x96\x16"
        b">\\\xe6\x85=\xe7\xbc\xaf\xbdN\xeb\xca8\x17mKQ^\xd3"
        b"\xb4\xd0\xc1$\x7f\x9d\xab+\x89`\x8cZ\x1b\x18o\xec\xdc\xbf\xd3"
        b'\xbb\xc3\xc3A\xf9\xc6\xcd"\xf7G\xda\x92:\x1dK\xb67'
        b"\xe5\xfak$\xdb\xc9\xd5\xfdg:\xd9N\xce\x8b\xd6\xd4\xf1^7\x0e\xeb"
        b"J\xf4\x8c\x0b\x99t\xf5\xa9)\xfb\x9d\xd6\xd5Y.t\r"
        b"\xf4-\x86\xd6\xa2\xf9@z\xd3YTo\x9a\xbew\x8a|\rqY\xa3\xbf\xde"
        b"ZG2\x1e\xc1>\xcb\xfb\xed\xb3V\xb2\xdf\xe6Y\xa5\xc0"
        b"\x88\x8e\x9b\x81-\n\x00\x00\x00\x00\xe0D1("
        b"\x8eN)\xcf\x99j|\xa9\xfb\xf1q`9j\x9d\xeb\xc6\xd1\x13&\x9bX\xef>_\xc6\xc6\xd3\xd3\xe3\xbdPS\x8f\x8f\xa7kG\x10G"
        b"\xeb\x87\xfa\x99\x1f\xe0\xb7iG\x9f\x86\xb5\x18\xbb\xb7\x8a]\\|\xfd\xc5\x8e\xe6\xd3\xca}@\\\x0e\x00\x00\x00"
        b"\x000lqt\xf4\x9cC}\x1f}\xf0\x9c#\xcd\x92\xce96\xe8\rm6R\xdf\xb0\x9fs\xb8N\xfbb!\xfc\xbc\x14\xf6Y\x06\xf9:\xa4"
        b"\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x1cK>\xde>\xdf\xb9\xf6\xc3\xec\xfc"
        b"}\xf0\xc7\xbe7\xf5m\xe6\xf2\xe7\xa7\xf6^\xee\xc9\xd6\xefG\xb3\x9d\x87\xcf\x9c\xed_\x1f&\xbf\xbe\xd4\xcet\xae"
        b'\xaf\xbf\x99\xc9~\xda\xdd\xaa/\xbf\xb7\xef.\xeen\xbf^2"\xff\x82\xfd\x07\xf7a\x8c\xa2\xd4>\x00\x00'
    )
    import gzip

    fn = get_test_tmp_file(suffix=".hdf")
    with open(fn, "wb") as f:
        f.write(gzip.decompress(raw_gzipped))

    dataset = HDFDataset(files=[fn])
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    print("tags:", reader.seq_tags)
    assert len(seq_lens) == reader.num_seqs
    assert_equal(reader.seq_tags, ["seq-0", "seq-1"])
    for i, seq_len in enumerate(seq_lens):
        assert reader.seq_lens[i]["data"] == seq_len
    assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == [n_dim]
    assert reader.data_dtype["data"] == "float32"


def test_SimpleHDFWriter_ndim1_var_len():
    # E.g. attention weights, shape (dec-time,enc-time) per seq.
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    writer = SimpleHDFWriter(filename=fn, dim=None, ndim=2, labels=None)
    dec_seq_lens1 = [11, 7, 5]
    enc_seq_lens1 = [13, 6, 8]
    batch1_data = numpy.random.normal(size=(len(dec_seq_lens1), max(dec_seq_lens1), max(enc_seq_lens1))).astype(
        "float32"
    )
    writer.insert_batch(
        inputs=batch1_data,
        seq_len={0: dec_seq_lens1, 1: enc_seq_lens1},
        seq_tag=["seq-%i" % i for i in range(len(dec_seq_lens1))],
    )
    dec_seq_lens2 = [10, 13, 3, 2]
    enc_seq_lens2 = [11, 13, 5, 4]
    batch2_data = numpy.random.normal(size=(len(dec_seq_lens2), max(dec_seq_lens2), max(enc_seq_lens2))).astype(
        "float32"
    )
    writer.insert_batch(
        inputs=batch2_data,
        seq_len={0: dec_seq_lens2, 1: enc_seq_lens2},
        seq_tag=["seq-%i" % (i + len(dec_seq_lens1)) for i in range(len(dec_seq_lens2))],
    )
    writer.close()
    dec_seq_lens = dec_seq_lens1 + dec_seq_lens2
    enc_seq_lens = enc_seq_lens1 + enc_seq_lens2

    dataset = HDFDataset(files=[fn])
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == []
    assert reader.data_dtype["data"] == "float32"
    assert "sizes" in reader.data_keys
    # sizes sparsity does not really matter...
    assert reader.data_dtype["sizes"] == "int32"
    assert list(reader.data_shape["sizes"]) == []
    assert len(dec_seq_lens) == len(enc_seq_lens) == reader.num_seqs
    for i, (dec_seq_len, enc_seq_len) in enumerate(zip(dec_seq_lens, enc_seq_lens)):
        assert reader.seq_lens[i]["data"] == dec_seq_len * enc_seq_len
        assert reader.seq_lens[i]["sizes"] == 2
        assert reader.data["sizes"][i].tolist() == [dec_seq_len, enc_seq_len], "got %r" % (reader.data["sizes"][i],)


def test_SimpleHDFWriter_extend_existing_file():
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    n_dim = 3
    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None)
    seq_lens = [2, 3]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens), max(seq_lens), n_dim)).astype("float32"),
        seq_len=seq_lens,
        seq_tag=["seq-%i" % i for i in range(len(seq_lens))],
    )
    writer.close()
    assert os.path.exists(fn)

    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None, extend_existing_file=True)
    seq_lens2 = [4, 3, 2]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens2), max(seq_lens2), n_dim)).astype("float32"),
        seq_len=seq_lens2,
        seq_tag=["seq-%i" % (i + len(seq_lens)) for i in range(len(seq_lens2))],
    )
    writer.close()
    seq_lens += seq_lens2

    dataset = HDFDataset(files=[fn])
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == [n_dim]
    assert reader.data_dtype["data"] == "float32"
    assert len(seq_lens) == reader.num_seqs
    for i, seq_len in enumerate(seq_lens):
        assert reader.seq_lens[i]["data"] == seq_len


@unittest.skip("unfinished...")
def test_SimpleHDFWriter_swmr():
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    rnd = numpy.random.RandomState(42)
    n_dim = 13
    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None, swmr=True)

    # TODO
    # As we directly want to read it, we must know all the seqs (and seq tags) in advance.
    # Not only that; we also would need all seq lens in advance, to have the offsets in the HDF.
    # (Or do we? We could extend MetaDataset to ignore missing.
    #  But then we would need to enforce a reload at the next epoch. Which we can do.)
    # In any case though, this would not support overwriting existing seqs,
    # but we need that...
    # How to solve this? Different HDF format? But not really possible even in principle in a nice way, or is it?
    # Or write to new HDF file for each epoch?
    # Once finished through whole dataset (all sub epochs), delete old and replace?
    # Then we also don't really need swmr.

    reader = HDFDataset(files=[fn])  # TODO use swmr?
    reader.init_seq_order(epoch=1)

    seq_lens = [11, 7, 5, 10, 13, 3, 2]
    # Construct seq tags in a way that the lengths vary.
    seq_tags = [
        ("seq-%i-" % i) + "".join(["abcdefghijk"[rnd.randint(0, 10)] for _ in range(rnd.randint(1, 50))])
        for i in range(len(seq_lens))
    ]
    seqs = numpy.random.normal(size=(len(seq_lens), max(seq_lens), n_dim)).astype("float32")
    for s, e in [(0, 3), (3, len(seq_lens))]:
        writer.insert_batch(inputs=seqs[s:e, : max(seq_lens[s:e])], seq_len=seq_lens[s:e], seq_tag=seq_tags[s:e])
        writer.flush()  # TODO when do we want it?

        # Now read.
        for seq_idx in range(s, e):
            reader.load_seqs(seq_idx, seq_idx + 1)
            # TODO read and check whether it is correct

    writer.close()  # Should not matter.


def test_SimpleHDFWriter_labels():
    fn = get_test_tmp_file(suffix=".hdf")
    os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
    n_dim = 3
    writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=[" ", "a", "b"])
    seq_lens = [2, 3]
    writer.insert_batch(
        inputs=numpy.random.normal(size=(len(seq_lens), max(seq_lens), n_dim)).astype("float32"),
        seq_len=seq_lens,
        seq_tag=["seq-%i" % i for i in range(len(seq_lens))],
    )
    writer.close()

    old_dataset = Old2018HDFDataset()
    old_dataset.add_file(fn)
    old_dataset.initialize()
    print("Old dataset:", old_dataset)
    print("Old dataset outputs:", old_dataset.num_outputs)
    print("Old dataset target keys:", old_dataset.target_keys)
    old_dataset.init_seq_order(epoch=1)
    old_dataset.load_seqs(0, 1)

    dataset = HDFDataset(files=[fn])
    print("Dataset:")
    print("  input:", dataset.num_inputs, "x", dataset.window)
    print("  output:", dataset.num_outputs)
    print(" ", dataset.len_info(fast=True) or "no info")
    print("Dataset keys:", dataset.get_data_keys())
    print("Dataset target keys:", dataset.get_target_list())
    print("Dataset labels:", ", ".join(f"{k!r}: {v[:3]}... len {len(v)}" for k, v in dataset.labels.items()) or "None")
    assert dataset.get_data_keys() == ["data"]
    assert dataset.get_target_list() == []

    assert dataset.labels["data"] == [" ", "a", "b"]
    reader = DatasetTestReader(dataset=dataset)
    reader.read_all()
    assert reader.data_keys == ["data"]
    assert reader.data_sparse["data"] is False
    assert list(reader.data_shape["data"]) == [n_dim]
    assert reader.data_dtype["data"] == "float32"
    assert len(seq_lens) == reader.num_seqs
    for i, seq_len in enumerate(seq_lens):
        assert reader.seq_lens[i]["data"] == seq_len


class Old2018HDFDataset(CachedDataset):
    """
    Copied and adapted from an early RETURNN version:
    2018-03-09: https://github.com/rwth-i6/returnn/blob/c2d8fed877022d1ac1bf68b801604733db51223e/HDFDataset.py

    (Not really fully functional here though... we just use it to test whether loading works.)

    Some version history:
    2015-01-05: https://github.com/rwth-i6/returnn/blob/3be0fe0906212d1ffdd93807c0a3854a38842eb8/Dataset.py
    2015-08-04: https://github.com/rwth-i6/returnn/blob/995e87184abc6e07256417cb533163ac0a7d7dd8/HDFDataset.py
    2018-03-09: https://github.com/rwth-i6/returnn/blob/c2d8fed877022d1ac1bf68b801604733db51223e/HDFDataset.py
    """

    def __init__(self, **kwargs):
        super(Old2018HDFDataset, self).__init__(cache_byte_size=100, **kwargs)
        self.files: List[str] = []
        self.file_start = [0]
        self.file_seq_start: List[List[int]] = []
        self.file_index: List[int] = []
        self.data_dtype: Dict[str, str] = {}
        self.data_sparse: Dict[str, bool] = {}

        # Copied from old base CachedDataset
        self._seq_lengths: List[Tuple[int, int]] = []  # uses real seq idx
        self.tags: List[str] = []  # uses real seq idx
        self.tag_idx: Dict[str, int] = {}  # map of tag -> real-seq-idx
        self.targets = {}
        self.target_keys = []

        # Copied from old base Dataset
        self._num_codesteps: Optional[int] = None  # Num output frames, could be different from input, seq2seq, ctc.

    def add_file(self, filename):
        """
        Setups data:
          self.seq_lengths
          self.file_index
          self.file_start
          self.file_seq_start
        Use load_seqs() to load the actual data.
        :type filename: str
        """
        fin = h5py.File(filename, "r")
        decode = lambda s: s if isinstance(s, str) else s.decode("utf-8")
        if "targets" in fin:
            self.labels = {
                k: [decode(item).split("\0")[0] for item in fin["targets/labels"][k][...].tolist()]
                for k in fin["targets/labels"]
            }
        if not self.labels:
            labels = [decode(item).split("\0")[0] for item in fin["labels"][...].tolist()]
            """:type: list[str]"""
            self.labels = {"classes": labels}
            assert len(self.labels["classes"]) == len(labels), (
                "expected " + str(len(self.labels["classes"])) + " got " + str(len(labels))
            )
        tags = [decode(item).split("\0")[0] for item in fin["seqTags"][...].tolist()]
        """ :type: list[str] """
        self.files.append(filename)
        if "times" in fin:
            if self.timestamps is None:
                self.timestamps = fin[attr_times][...]
            else:
                self.timestamps = numpy.concatenate(
                    [self.timestamps, fin[attr_times][...]], axis=0
                )  # .extend(fin[attr_times][...].tolist())
        seq_lengths = fin[attr_seqLengths][...]
        if "targets" in fin:
            self.target_keys = sorted(fin["targets/labels"].keys())
        else:
            self.target_keys = ["classes"]

        if len(seq_lengths.shape) == 1:
            seq_lengths = numpy.array(zip(*[seq_lengths.tolist() for i in range(len(self.target_keys) + 1)]))

        seq_start = [numpy.zeros((seq_lengths.shape[1],), "int64")]
        if not self._seq_start:
            self._seq_start = [numpy.zeros((seq_lengths.shape[1],), "int64")]
        for l in seq_lengths:
            self._seq_lengths.append(numpy.array(l))
            seq_start.append(seq_start[-1] + l)
        self.tags += tags
        self.file_seq_start.append(seq_start)
        nseqs = len(seq_start) - 1
        for i in range(nseqs):
            self.tag_idx[tags[i]] = i + self._num_seqs
        self._num_seqs += nseqs
        self.file_index.extend([len(self.files) - 1] * nseqs)
        self.file_start.append(self.file_start[-1] + nseqs)
        self._num_timesteps += sum([s[0] for s in seq_lengths])
        if self._num_codesteps is None:
            self._num_codesteps = [0 for i in range(1, len(seq_lengths[0]))]
        for i in range(1, len(seq_lengths[0])):
            self._num_codesteps[i - 1] += sum([s[i] for s in seq_lengths])
        if len(fin["inputs"].shape) == 1:  # sparse
            num_inputs = [fin.attrs[attr_inputPattSize], 1]
        else:
            num_inputs = [fin["inputs"].shape[1], len(fin["inputs"].shape)]  # fin.attrs[attr_inputPattSize]
        if self.num_inputs == 0:
            self.num_inputs = num_inputs[0]
        assert self.num_inputs == num_inputs[0], "wrong input dimension in file %s (expected %s got %s)" % (
            filename,
            self.num_inputs,
            num_inputs[0],
        )
        if "targets/size" in fin:
            num_outputs = {
                k: [fin["targets/size"].attrs[k], len(fin["targets/data"][k].shape)] for k in fin["targets/size"].attrs
            }
        else:
            num_outputs = {"classes": fin.attrs["numLabels"]}
        num_outputs["data"] = num_inputs
        if not self.num_outputs:
            self.num_outputs = num_outputs
        assert self.num_outputs == num_outputs, "wrong dimensions in file %s (expected %s got %s)" % (
            filename,
            self.num_outputs,
            num_outputs,
        )
        if "targets" in fin:
            for name in fin["targets/data"]:
                tdim = 1 if len(fin["targets/data"][name].shape) == 1 else fin["targets/data"][name].shape[1]
                self.data_dtype[name] = str(fin["targets/data"][name].dtype) if tdim > 1 else "int32"
                self.targets[name] = None
        else:
            self.targets = {"classes": numpy.zeros((self._num_timesteps,))}
            self.data_dtype["classes"] = "int32"
        self.data_dtype["data"] = fin["inputs"].dtype
        assert len(self.target_keys) == len(self._seq_lengths[0]) - 1
        fin.close()

    def _load_seqs(self, start, end):
        """
        Load data sequences.
        As a side effect, will modify / fill-up:
          self.alloc_intervals
          self.targets
          self.chars

        :param int start: start sorted seq idx
        :param int end: end sorted seq idx
        """
        assert start < self.num_seqs
        assert end <= self.num_seqs
        selection = self.insert_alloc_interval(start, end)
        assert len(selection) <= end - start, (
            "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end - start) + ")"
        )
        file_info: List[List[int]] = [[] for l in range(len(self.files))]
        # file_info[i] is (sorted seq idx from selection, real seq idx)
        for idc in selection:
            ids = self._seq_index[idc]
            file_info[self.file_index[ids]].append((idc, ids))
            self.preload_set.add(ids)
        for i in range(len(self.files)):
            if len(file_info[i]) == 0:
                continue
            print("loading file %d/%d" % (i + 1, len(self.files)), self.files[i], file=log.v4)
            fin = h5py.File(self.files[i], "r")
            for idc, ids in file_info[i]:
                s = ids - self.file_start[i]
                p = self.file_seq_start[i][s]
                l = self._seq_lengths[ids]
                if "targets" in fin:
                    for k in fin["targets/data"]:
                        if self.targets[k] is None:
                            self.targets[k] = numpy.zeros((self._num_codesteps[self.target_keys.index(k)],)) - 1
                        ldx = self.target_keys.index(k) + 1
                        self.targets[k][self.get_seq_start(idc)[ldx] : self.get_seq_start(idc)[ldx] + l[ldx]] = fin[
                            "targets/data/" + k
                        ][p[ldx] : p[ldx] + l[ldx]]
                self._set_alloc_intervals_data(idc, data=fin["inputs"][p[0] : p[0] + l[0]][...])
            fin.close()
        gc.collect()
        assert self.is_cached(start, end)

    def _get_seq_length_by_real_idx(self, real_seq_idx):
        return self._seq_lengths[real_seq_idx]

    def get_tag(self, sorted_seq_idx):
        ids = self._seq_index[self._index_map[sorted_seq_idx]]
        return self.tags[ids]

    def is_data_sparse(self, key):
        if key in self.num_outputs:
            return self.num_outputs[key][1] == 1
        if self.get_data_dtype(key).startswith("int"):
            return True
        return False

    def get_data_dtype(self, key):
        return self.data_dtype[key]

    def len_info(self):
        return ", ".join(["HDF dataset", "sequences: %i" % self.num_seqs, "frames: %i" % self.get_num_timesteps()])


def dummy_iter_dataset(dataset: Dataset) -> int:
    """
    :param Dataset dataset:
    :return: num seqs
    """
    dataset.init_seq_order(epoch=1)
    data_keys = dataset.get_data_keys()
    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        for key in data_keys:
            dataset.get_data(seq_idx=seq_idx, key=key)
            dataset.get_tag(seq_idx)
        seq_idx += 1
    print("Iterated through %r, num seqs %i" % (dataset, seq_idx))
    return seq_idx


def test_hdf_simple_iter():
    hdf_fn = generate_hdf_from_dummy()
    dataset = HDFDataset(files=[hdf_fn])
    dataset.initialize()
    dummy_iter_dataset(dataset)


def test_hdf_simple_iter_cached():
    hdf_fn = generate_hdf_from_dummy()
    dataset = HDFDataset(files=[hdf_fn], cache_byte_size=100)
    dataset.initialize()
    dummy_iter_dataset(dataset)


def test_rnn_getCacheByteSizes_zero():
    from returnn.config import Config

    config = Config({"cache_size": "0"})
    import returnn.__main__ as rnn

    rnn.config = config
    sizes = rnn.get_cache_byte_sizes()
    assert len(sizes) == 3
    assert all([s == 0 for s in sizes])


def test_rnn_initData():
    hdf_fn = generate_hdf_from_dummy()
    from returnn.config import Config

    config = Config({"cache_size": "0", "train": hdf_fn, "dev": hdf_fn})
    import returnn.__main__ as rnn

    rnn.config = config
    rnn.init_data()
    train, dev = rnn.train_data, rnn.dev_data
    assert train and dev
    assert isinstance(train, HDFDataset)
    assert isinstance(dev, HDFDataset)
    assert train.cache_byte_size_total_limit == dev.cache_byte_size_total_limit == 0
    assert train.cache_byte_size_limit_at_start == dev.cache_byte_size_limit_at_start == 0


def test_hdf_no_cache_iter():
    hdf_fn = generate_hdf_from_dummy()
    dataset = HDFDataset(files=[hdf_fn])
    dataset.initialize()
    assert dataset.cache_byte_size_limit_at_start == 0
    assert dataset.cache_byte_size_total_limit == 0

    class DummyCallback:
        def __init__(self):
            self.was_called = False

        def __call__(self, *args, **kwargs):
            print("DummyCallback called!")
            self.was_called = True

    assert hasattr(dataset, "_preload_seqs")
    dataset._preload_seqs = DummyCallback()

    dummy_iter_dataset(dataset)
    import time

    time.sleep(1)  # maybe threads are running in background...
    assert not dataset._preload_seqs.was_called


def test_hdf_data_short_int_dtype():
    from returnn.datasets.generating import StaticDataset

    dataset = StaticDataset(
        [{"data": numpy.array([1, 2, 3], dtype="uint8"), "classes": numpy.array([-1, 5], dtype="int16")}],
        output_dim={"data": (255, 1), "classes": (10, 1)},
    )
    orig_data_dtype = dataset.get_data_dtype("data")
    orig_classes_dtype = dataset.get_data_dtype("classes")
    assert orig_data_dtype == "uint8" and orig_classes_dtype == "int16"

    hdf_fn = get_test_tmp_file(suffix=".hdf")
    hdf_writer = HDFDatasetWriter(filename=hdf_fn)
    hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
    hdf_writer.close()

    hdf_dataset = HDFDataset(files=[hdf_fn])
    hdf_dataset.initialize()
    hdf_dataset.init_seq_order(epoch=1)
    hdf_data_dtype = hdf_dataset.get_data_dtype("data")
    hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
    assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
    hdf_data_dim = hdf_dataset.get_data_dim("data")
    hdf_classes_dim = hdf_dataset.get_data_dim("classes")
    assert hdf_data_dim == 255 and hdf_classes_dim == 10
    hdf_data_shape = hdf_dataset.get_data_shape("data")
    hdf_classes_shape = hdf_dataset.get_data_shape("classes")
    assert hdf_data_shape == [] and hdf_classes_shape == []
    hdf_dataset.load_seqs(0, 1)
    hdf_data_data = hdf_dataset.get_data(0, "data")
    hdf_data_classes = hdf_dataset.get_data(0, "classes")
    assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_hdf_data_target_int32():
    from returnn.datasets.generating import StaticDataset

    dataset = StaticDataset(
        [
            {
                "data": numpy.array([1, 2, 3], dtype="uint8"),
                "classes": numpy.array([2147483647, 2147483646, 2147483645], dtype="int32"),
            }
        ],
        output_dim={"data": (255, 1), "classes": (10, 1)},
    )
    dataset.initialize()
    dataset.init_seq_order(epoch=0)
    dataset.load_seqs(0, 1)
    orig_classes_dtype = dataset.get_data_dtype("classes")
    orig_classes_seq = dataset.get_data(0, "classes")
    assert orig_classes_seq.shape == (3,) and orig_classes_seq[0] == 2147483647
    assert orig_classes_seq.dtype == orig_classes_dtype == "int32"

    hdf_fn = get_test_tmp_file(suffix=".hdf")
    hdf_writer = HDFDatasetWriter(filename=hdf_fn)
    hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
    hdf_writer.close()

    hdf_dataset = HDFDataset(files=[hdf_fn])
    hdf_dataset.initialize()
    hdf_dataset.init_seq_order(epoch=1)
    hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
    assert hdf_classes_dtype == orig_classes_dtype
    hdf_classes_shape = hdf_dataset.get_data_shape("classes")
    assert hdf_classes_shape == []
    hdf_dataset.load_seqs(0, 1)
    hdf_data_classes = hdf_dataset.get_data(0, "classes")
    assert hdf_data_classes.dtype == orig_classes_dtype
    assert all(hdf_data_classes == orig_classes_seq)


def test_hdf_target_float_dtype():
    from returnn.datasets.generating import StaticDataset

    dataset = StaticDataset(
        [{"data": numpy.array([1, 2, 3], dtype="float32"), "classes": numpy.array([-1, 5], dtype="float32")}],
        output_dim={"data": (1, 1), "classes": (1, 1)},
    )
    orig_data_dtype = dataset.get_data_dtype("data")
    orig_classes_dtype = dataset.get_data_dtype("classes")
    assert orig_data_dtype == "float32" and orig_classes_dtype == "float32"

    hdf_fn = get_test_tmp_file(suffix=".hdf")
    hdf_writer = HDFDatasetWriter(filename=hdf_fn)
    hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
    hdf_writer.close()

    hdf_dataset = HDFDataset(files=[hdf_fn])
    hdf_dataset.initialize()
    hdf_dataset.init_seq_order(epoch=1)
    hdf_data_dtype = hdf_dataset.get_data_dtype("data")
    hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
    assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
    hdf_data_dim = hdf_dataset.get_data_dim("data")
    hdf_classes_dim = hdf_dataset.get_data_dim("classes")
    assert hdf_data_dim == 1 and hdf_classes_dim == 1
    hdf_data_shape = hdf_dataset.get_data_shape("data")
    hdf_classes_shape = hdf_dataset.get_data_shape("classes")
    assert hdf_data_shape == [] and hdf_classes_shape == []
    hdf_dataset.load_seqs(0, 1)
    hdf_data_data = hdf_dataset.get_data(0, "data")
    hdf_data_classes = hdf_dataset.get_data(0, "classes")
    assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_hdf_target_float_dense():
    from returnn.datasets.generating import StaticDataset

    dataset = StaticDataset(
        [
            {
                "data": numpy.array([[1, 2, 3], [2, 3, 4]], dtype="float32"),
                "classes": numpy.array([[-1, 5], [-2, 4], [-3, 2]], dtype="float32"),
            }
        ]
    )
    orig_data_dtype = dataset.get_data_dtype("data")
    orig_classes_dtype = dataset.get_data_dtype("classes")
    assert orig_data_dtype == "float32" and orig_classes_dtype == "float32"
    orig_data_shape = dataset.get_data_shape("data")
    orig_classes_shape = dataset.get_data_shape("classes")
    assert orig_data_shape == [3] and orig_classes_shape == [2]

    hdf_fn = get_test_tmp_file(suffix=".hdf")
    hdf_writer = HDFDatasetWriter(filename=hdf_fn)
    hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
    hdf_writer.close()

    hdf_dataset = HDFDataset(files=[hdf_fn])
    hdf_dataset.initialize()
    hdf_dataset.init_seq_order(epoch=1)
    hdf_data_dtype = hdf_dataset.get_data_dtype("data")
    hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
    assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
    hdf_data_dim = hdf_dataset.get_data_dim("data")
    hdf_classes_dim = hdf_dataset.get_data_dim("classes")
    assert hdf_data_dim == orig_data_shape[-1] and hdf_classes_dim == orig_classes_shape[-1]
    hdf_data_shape = hdf_dataset.get_data_shape("data")
    hdf_classes_shape = hdf_dataset.get_data_shape("classes")
    assert hdf_data_shape == orig_data_shape and hdf_classes_shape == orig_classes_shape
    hdf_dataset.load_seqs(0, 1)
    hdf_data_data = hdf_dataset.get_data(0, "data")
    hdf_data_classes = hdf_dataset.get_data(0, "classes")
    assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_HDFDataset_no_cache_efficiency():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset = HDFDataset(files=[hdf_fn], cache_byte_size=0)
    hdf_dataset.initialize()
    hdf_dataset.init_seq_order(epoch=1)
    hdf_dataset.load_seqs(0, 1)
    hdf_dataset.load_seqs(0, 2)
    hdf_dataset.load_seqs(0, 5)
    hdf_dataset.load_seqs(1, 2)
    hdf_dataset.load_seqs(1, 3)
    hdf_dataset.load_seqs(1, 7)
    hdf_dataset.load_seqs(3, 7)
    hdf_dataset.load_seqs(4, 10)
    # TODO... check alloc intervals etc


def test_HDFDataset_pickle():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    ds = HDFDataset(files=[hdf_fn], cache_byte_size=0)
    ds.initialize()
    ds.init_seq_order(epoch=1)

    import pickle

    ds_copy = pickle.loads(pickle.dumps(ds))
    assert ds_copy.epoch == 1
    num_seqs = dummy_iter_dataset(ds)
    num_seqs_copy = dummy_iter_dataset(ds_copy)
    assert num_seqs == num_seqs_copy


def test_HDFDataset_deepcopy():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    ds = HDFDataset(files=[hdf_fn], cache_byte_size=0)
    ds.initialize()
    ds.init_seq_order(epoch=1)

    import copy

    ds_copy = copy.deepcopy(ds)
    assert ds_copy.epoch == 1
    num_seqs = dummy_iter_dataset(ds)
    num_seqs_copy = dummy_iter_dataset(ds_copy)
    assert num_seqs == num_seqs_copy


def test_siamese_triplet_sampling():
    datasets_path = generate_dummy_hdf(3)
    dataset = SiameseHDFDataset(input_stream_name="features", seq_label_stream="classes", files=datasets_path)

    dataset.initialize()
    for iter in range(1, 31):
        print("Initializing triplets... iteration %d" % iter)
        dataset.init_seq_order(epoch=iter)

        triplets = dataset.curr_epoch_triplets
        anchor_seq_names = [dataset.all_seq_names[id[0]] for id in triplets]
        same_class_seq_names = [dataset.all_seq_names[id[1]] for id in triplets]
        diff_class_seq_names = [dataset.all_seq_names[id[2]] for id in triplets]

        anchor_class = [dataset.seq_to_target[seq_id] for seq_id in anchor_seq_names]
        same_class = [dataset.seq_to_target[seq_id] for seq_id in same_class_seq_names]
        diff_class = [dataset.seq_to_target[seq_id] for seq_id in diff_class_seq_names]

        print("Testing pair sequences to belong to the same class...")
        assert all(ac == same_class[id] for id, ac in enumerate(anchor_class))
        print("Testing third element in a triplet to belong to a different class...")
        assert all(ac != diff_class[id] for id, ac in enumerate(anchor_class))
        print("------------------------------------------------------")

    print("Deleting temporary files...")
    for path in datasets_path:
        os.remove(path)
    print("Done.")


def test_siamese_collect_single_seq():
    datasets_path = generate_dummy_hdf(3)
    dataset = SiameseHDFDataset(input_stream_name="features", seq_label_stream="classes", files=datasets_path)

    dataset.initialize()
    dataset.init_seq_order(epoch=1)

    random = np.random.RandomState()
    seq_idx = random.randint(low=0, high=len(dataset.seq_name_to_idx))
    dataset_seq = dataset._collect_single_seq(seq_idx)
    print("Verify that single sequence consists of a triplet...")
    print(dataset_seq.features.keys())

    print("Deleting temporary files...")
    for path in datasets_path:
        os.remove(path)
    print("Done.")


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
