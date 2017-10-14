
from __future__ import print_function

from Dataset import Dataset, DatasetSeq, convert_data_dims
from CachedDataset2 import CachedDataset2
from Util import class_idx_seq_to_1_of_k
from Log import log
import numpy


class GeneratingDataset(Dataset):

  _input_classes = None
  _output_classes = None

  def __init__(self, input_dim, output_dim, num_seqs=float("inf"), fixed_random_seed=None, **kwargs):
    """
    :param int input_dim:
    :param int|dict[str,int|(int,int)|dict] output_dim:
    :param int|float num_seqs:
    :param int fixed_random_seed:
    """
    super(GeneratingDataset, self).__init__(**kwargs)
    assert self.shuffle_frames_of_nseqs == 0

    self.num_inputs = input_dim
    output_dim = convert_data_dims(output_dim)
    if "data" not in output_dim:
      output_dim["data"] = [input_dim, 2]  # not sparse
    self.num_outputs = output_dim
    self.expected_load_seq_start = 0
    self._num_seqs = num_seqs
    self.random = numpy.random.RandomState(1)
    self.fixed_random_seed = fixed_random_seed  # useful when used as eval dataset

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param seq_list: predefined order. doesn't make sense here
    This is called when we start a new epoch, or at initialization.
    """
    super(GeneratingDataset, self).init_seq_order(epoch=epoch)
    assert not seq_list, "predefined order doesn't make sense for %s" % self.__class__.__name__
    self.random.seed(self.fixed_random_seed or epoch or 1)
    self._num_timesteps = 0
    self.reached_final_seq = False
    self.expected_load_seq_start = 0
    self.added_data = []; " :type: list[DatasetSeq] "
    return True

  def _cleanup_old_seqs(self, seq_idx_end):
    i = 0
    while i < len(self.added_data):
      if self.added_data[i].seq_idx >= seq_idx_end:
        break
      i += 1
    del self.added_data[:i]

  def _check_loaded_seq_idx(self, seq_idx):
    if not self.added_data:
      raise Exception("no data loaded yet")
    start_loaded_seq_idx = self.added_data[0].seq_idx
    end_loaded_seq_idx = self.added_data[-1].seq_idx
    if seq_idx < start_loaded_seq_idx or seq_idx > end_loaded_seq_idx:
      raise Exception("seq_idx %i not in loaded seqs range [%i,%i]" % (
        seq_idx, start_loaded_seq_idx, end_loaded_seq_idx))

  def _get_seq(self, seq_idx):
    for data in self.added_data:
      if data.seq_idx == seq_idx:
        return data
    return None

  def is_cached(self, start, end):
    # Always False, to force that we call self._load_seqs().
    # This is important for our buffer management.
    return False

  def _load_seqs(self, start, end):
    """
    :param int start: inclusive seq idx start
    :param int end: exclusive seq idx end
    """
    # We expect that start increase monotonic on each call
    # for not-yet-loaded data.
    # This will already be called with _load_seqs_superset indices.
    assert start >= self.expected_load_seq_start
    if start > self.expected_load_seq_start:
      # Cleanup old data.
      self._cleanup_old_seqs(start)
      self.expected_load_seq_start = start
    if self.added_data:
      start = max(self.added_data[-1].seq_idx + 1, start)
    if end > self.num_seqs:
      end = self.num_seqs
    if end >= self.num_seqs:
      self.reached_final_seq = True
    seqs = [self.generate_seq(seq_idx=seq_idx) for seq_idx in range(start, end)]
    if self.window > 1:
      for seq in seqs:
        seq.features = self.sliding_window(seq.features)
    self._num_timesteps += sum([seq.num_frames for seq in seqs])
    self.added_data += seqs

  def generate_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    raise NotImplementedError

  def _shuffle_frames_in_seqs(self, start, end):
    assert False, "Shuffling in GeneratingDataset does not make sense."

  def get_num_timesteps(self):
    assert self.reached_final_seq
    return self._num_timesteps

  @property
  def num_seqs(self):
    return self._num_seqs

  def get_seq_length(self, sorted_seq_idx):
    # get_seq_length() can be called before the seq is loaded via load_seqs().
    # Thus, we just call load_seqs() ourselves here.
    assert sorted_seq_idx >= self.expected_load_seq_start
    self.load_seqs(self.expected_load_seq_start, sorted_seq_idx + 1)
    return self._get_seq(sorted_seq_idx).num_frames

  def get_input_data(self, sorted_seq_idx):
    self._check_loaded_seq_idx(sorted_seq_idx)
    return self._get_seq(sorted_seq_idx).features

  def get_targets(self, target, sorted_seq_idx):
    self._check_loaded_seq_idx(sorted_seq_idx)
    return self._get_seq(sorted_seq_idx).targets[target]

  def get_ctc_targets(self, sorted_seq_idx):
    self._check_loaded_seq_idx(sorted_seq_idx)
    assert self._get_seq(sorted_seq_idx).ctc_targets

  def get_tag(self, sorted_seq_idx):
    self._check_loaded_seq_idx(sorted_seq_idx)
    return self._get_seq(sorted_seq_idx).seq_tag


class Task12AXDataset(GeneratingDataset):
  """
  12AX memory task.
  This is a simple memory task where there is an outer loop and an inner loop.
  Description here: http://psych.colorado.edu/~oreilly/pubs-abstr.html#OReillyFrank06
  """

  _input_classes = "123ABCXYZ"
  _output_classes = "LR"

  def __init__(self, **kwargs):
    super(Task12AXDataset, self).__init__(
      input_dim=len(self._input_classes),
      output_dim=len(self._output_classes),
      **kwargs)

  def get_random_seq_len(self):
    return self.random.randint(10, 100)

  def generate_input_seq(self, seq_len):
    """
    Somewhat made up probability distribution.
    Try to make in a way that at least some "R" will occur in the output seq.
    Otherwise, "R"s are really rare.
    """
    seq = self.random.choice(["", "1", "2"])
    while len(seq) < seq_len:
      if self.random.uniform() < 0.5:
        seq += self.random.choice(list("12"))
      if self.random.uniform() < 0.9:
        seq += self.random.choice(["AX", "BY"])
      while self.random.uniform() < 0.5:
        seq += self.random.choice(list(self._input_classes))
    return list(map(self._input_classes.index, seq[:seq_len]))

  @classmethod
  def make_output_seq(cls, input_seq):
    """
    :type input_seq: list[int]
    :rtype: list[int]
    """
    outer_state = ""
    inner_state = ""
    input_classes = cls._input_classes
    output_seq_str = ""
    for i in input_seq:
      c = input_classes[i]
      o = "L"
      if c in "12":
        outer_state = c
      elif c in "AB":
        inner_state = c
      elif c in "XY":
        if outer_state + inner_state + c in ["1AX", "2BY"]:
          o = "R"
        inner_state = ""
      # Ignore other cases, "3CZ".
      output_seq_str += o
    return list(map(cls._output_classes.index, output_seq_str))

  def estimate_output_class_priors(self, num_trials, seq_len=10):
    """
    :type num_trials: int
    :rtype: (float, float)
    """
    count_l, count_r = 0, 0
    for i in range(num_trials):
      input_seq = self.generate_input_seq(seq_len)
      output_seq = self.make_output_seq(input_seq)
      count_l += output_seq.count(0)
      count_r += output_seq.count(1)
    return float(count_l) / (num_trials * seq_len), float(count_r) / (num_trials * seq_len)

  def generate_seq(self, seq_idx):
    seq_len = self.get_random_seq_len()
    input_seq = self.generate_input_seq(seq_len)
    output_seq = self.make_output_seq(input_seq)
    features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
    targets = numpy.array(output_seq)
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskEpisodicCopyDataset(GeneratingDataset):
  """
  Episodic Copy memory task.
  This is a simple memory task where we need to remember a sequence.
  Described in: http://arxiv.org/abs/1511.06464
  Also tested for Associative LSTMs.
  This is a variant where the lengths are random, both for the chars and for blanks.
  """

  # Blank, delimiter and some chars.
  _input_classes = " .01234567"
  _output_classes = _input_classes

  def __init__(self, **kwargs):
    super(TaskEpisodicCopyDataset, self).__init__(
      input_dim=len(self._input_classes),
      output_dim=len(self._output_classes),
      **kwargs)

  def generate_input_seq(self):
    seq = ""
    # Start with random chars.
    rnd_char_len = self.random.randint(1, 10)
    seq += "".join([self.random.choice(list(self._input_classes[2:]))
                    for i in range(rnd_char_len)])
    blank_len = self.random.randint(1, 100)
    seq += " " * blank_len  # blanks
    seq += "."  # 1 delim
    seq += "." * (rnd_char_len + 1)  # we wait for the outputs + 1 delim
    return list(map(self._input_classes.index, seq))

  @classmethod
  def make_output_seq(cls, input_seq):
    """
    :type input_seq: list[int]
    :rtype: list[int]
    """
    input_classes = cls._input_classes
    input_mem = ""
    output_seq_str = ""
    state = 0
    for i in input_seq:
      c = input_classes[i]
      if state == 0:
        output_seq_str += " "
        if c == " ": pass  # just ignore
        elif c == ".": state = 1  # start with recall now
        else: input_mem += c
      else:  # recall from memory
        # Ignore input.
        if not input_mem:
          output_seq_str += "."
        else:
          output_seq_str += input_mem[:1]
          input_mem = input_mem[1:]
    return list(map(cls._output_classes.index, output_seq_str))

  def generate_seq(self, seq_idx):
    input_seq = self.generate_input_seq()
    output_seq = self.make_output_seq(input_seq)
    features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
    targets = numpy.array(output_seq)
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskXmlModelingDataset(GeneratingDataset):
  """
  XML modeling memory task.
  This is a memory task where we need to remember a stack.
  Defined in Jozefowicz et al. (2015).
  Also tested for Associative LSTMs.
  """

  # Blank, XML-tags and some chars.
  _input_classes = " <>/abcdefgh"
  _output_classes = _input_classes

  def __init__(self, limit_stack_depth=4, **kwargs):
    super(TaskXmlModelingDataset, self).__init__(
      input_dim=len(self._input_classes),
      output_dim=len(self._output_classes),
      **kwargs)
    self.limit_stack_depth = limit_stack_depth

  def generate_input_seq(self):
    # Because this is a prediction task, start with blank,
    # and the output seq should predict the next char after the blank.
    seq = " "
    xml_stack = []
    while True:
      if not xml_stack or (len(xml_stack) < self.limit_stack_depth and self.random.rand() > 0.6):
        tag_len = self.random.randint(1, 10)
        tag = "".join([self.random.choice(list(self._input_classes[4:]))
                       for i in range(tag_len)])
        seq += "<%s>" % tag
        xml_stack += [tag]
      else:
        seq += "</%s>" % xml_stack.pop()
      if not xml_stack and self.random.rand() > 0.2:
        break
    return list(map(self._input_classes.index, seq))

  @classmethod
  def make_output_seq(cls, input_seq):
    """
    :type input_seq: list[int]
    :rtype: list[int]
    """
    input_classes = cls._input_classes
    input_seq_str = "".join(cls._input_classes[i] for i in input_seq)
    xml_stack = []
    output_seq_str = ""
    state = 0
    for c in input_seq_str:
      if c in " >":
        output_seq_str += "<"  # We expect an open char.
        assert state != 1, repr(input_seq_str)
        state = 1  # expect beginning of tag
      elif state == 1:  # in beginning of tag
        output_seq_str += " "  # We don't know yet.
        assert c == "<", repr(input_seq_str)
        state = 2
      elif state == 2:  # first char in tag
        if c == "/":
          assert xml_stack, repr(input_seq_str)
          output_seq_str += xml_stack[-1][0]
          xml_stack[-1] = xml_stack[-1][1:]
          state = 4  # closing tag
        else:  # opening tag
          output_seq_str += " "  # We don't know yet.
          assert c not in " <>/", repr(input_seq_str)
          state = 3
          xml_stack += [c]
      elif state == 3:  # opening tag
        output_seq_str += " "  # We don't know.
        xml_stack[-1] += c
      elif state == 4:  # closing tag
        assert xml_stack, repr(input_seq_str)
        if not xml_stack[-1]:
          output_seq_str += ">"
          xml_stack.pop()
          state = 0
        else:
          output_seq_str += xml_stack[-1][0]
          xml_stack[-1] = xml_stack[-1][1:]
      else:
        assert False, "invalid state %i. input %r" % (state, input_seq_str)
    return list(map(cls._output_classes.index, output_seq_str))

  def generate_seq(self, seq_idx):
    input_seq = self.generate_input_seq()
    output_seq = self.make_output_seq(input_seq)
    features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
    targets = numpy.array(output_seq)
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class TaskVariableAssignmentDataset(GeneratingDataset):
  """
  Variable Assignment memory task.
  This is a memory task to test for key-value retrieval.
  Defined in Associative LSTM paper.
  """

  # Blank/Delim/End, Store/Query, and some chars for key/value.
  _input_classes = " ,.SQ()abcdefgh"
  _output_classes = _input_classes

  def __init__(self, **kwargs):
    super(TaskVariableAssignmentDataset, self).__init__(
      input_dim=len(self._input_classes),
      output_dim=len(self._output_classes),
      **kwargs)

  def generate_input_seq(self):
    seq = ""
    from collections import OrderedDict
    store = OrderedDict()
    # First the assignments.
    num_assignments = self.random.randint(1, 5)
    for i in range(num_assignments):
      key_len = self.random.randint(2, 5)
      while True:  # find unique key
        key = "".join([self.random.choice(list(self._input_classes[7:]))
                       for i in range(key_len)])
        if key not in store: break
      value_len = self.random.randint(1, 2)
      value = "".join([self.random.choice(list(self._input_classes[7:]))
                       for i in range(value_len)])
      if seq: seq += ","
      seq += "S(%s,%s)" % (key, value)
      store[key] = value
    # Now one query.
    key = self.random.choice(store.keys())
    value = store[key]
    seq += ",Q(%s)" % key
    seq += "%s." % value
    return list(map(self._input_classes.index, seq))

  @classmethod
  def make_output_seq(cls, input_seq):
    """
    :type input_seq: list[int]
    :rtype: list[int]
    """
    input_classes = cls._input_classes
    input_seq_str = "".join(cls._input_classes[i] for i in input_seq)
    store = {}
    key, value = "", ""
    output_seq_str = ""
    state = 0
    for c in input_seq_str:
      if state == 0:
        key = ""
        if c == "S": state = 1  # store
        elif c == "Q": state = 2  # query
        elif c in " ,": pass  # can be ignored
        else: assert False, "c %r in %r" % (c, input_seq_str)
        output_seq_str += " "
      elif state == 1:  # store
        assert c == "(", repr(input_seq_str)
        state = 1.1
        output_seq_str += " "
      elif state == 1.1:  # store.key
        if c == ",":
          assert key
          value = ""
          state = 1.5  # store.value
        else:
          assert c not in " .,SQ()", repr(input_seq_str)
          key += c
        output_seq_str += " "
      elif state == 1.5:  # store.value
        if c == ")":
          assert value
          store[key] = value
          state = 0
        else:
          assert c not in " .,SQ()", repr(input_seq_str)
          value += c
        output_seq_str += " "
      elif state == 2:  # query
        assert c == "(", repr(input_seq_str)
        state = 2.1
        output_seq_str += " "
      elif state == 2.1:  # query.key
        if c == ")":
          value = store[key]
          output_seq_str += value[0]
          value = value[1:]
          state = 2.5
        else:
          assert c not in " .,SQ()", repr(input_seq_str)
          key += c
          output_seq_str += " "
      elif state == 2.5:  # query result
        assert c not in " .,SQ()", repr(input_seq_str)
        if value:
          output_seq_str += value[0]
          value = value[1:]
        else:
          output_seq_str += "."
          state = 2.6
      elif state == 2.6:  # query result end
        assert c == ".", repr(input_seq_str)
        output_seq_str += " "
      else:
        assert False, "invalid state %i, input %r" % (state, input_seq_str)
    return list(map(cls._output_classes.index, output_seq_str))

  def generate_seq(self, seq_idx):
    input_seq = self.generate_input_seq()
    output_seq = self.make_output_seq(input_seq)
    features = class_idx_seq_to_1_of_k(input_seq, num_classes=len(self._input_classes))
    targets = numpy.array(output_seq)
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class DummyDataset(GeneratingDataset):

  def __init__(self, input_dim, output_dim, num_seqs, seq_len=2,
               input_max_value=10.0, input_shift=None, input_scale=None, **kwargs):
    super(DummyDataset, self).__init__(input_dim=input_dim, output_dim=output_dim, num_seqs=num_seqs, **kwargs)
    self.seq_len = seq_len
    self.input_max_value = input_max_value
    if input_shift is None: input_shift = -input_max_value / 2.0
    self.input_shift = input_shift
    if input_scale is None: input_scale = 1.0 / self.input_max_value
    self.input_scale = input_scale

  def generate_seq(self, seq_idx):
    seq_len = self.seq_len
    i1 = seq_idx
    i2 = i1 + seq_len * self.num_inputs
    features = numpy.array([((i % self.input_max_value) + self.input_shift) * self.input_scale
                            for i in range(i1, i2)]).reshape((seq_len, self.num_inputs))
    i1, i2 = i2, i2 + seq_len
    targets = numpy.array([i % self.num_outputs["classes"][0]
                           for i in range(i1, i2)])
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


class StaticDataset(GeneratingDataset):

  def __init__(self, data, target_list=None, output_dim=None, input_dim=None, **kwargs):
    """
    :type data: list[dict[str,numpy.ndarray]]
    """
    assert len(data) > 0
    self.data = data
    num_seqs = len(data)
    first_data = data[0]
    assert "data" in first_data  # input
    if target_list is None:
      target_list = []
      for target in first_data.keys():
        if target == "data": continue
        target_list.append(target)
    else:
      for target in target_list:
        assert target in first_data
    self.target_list = target_list

    if output_dim is None:
      output_dim = {}
    output_dim = convert_data_dims(output_dim)

    first_data_input = first_data["data"]
    assert len(first_data_input.shape) <= 2  # (time[,dim])
    if input_dim is None:
      if "data" in output_dim:
        input_dim = output_dim["data"][0]
      else:
        input_dim = first_data_input.shape[1]

    for target in target_list:
      first_data_output = first_data[target]
      assert len(first_data_output.shape) <= 2  # (time[,dim])
      if target in output_dim:
        assert output_dim[target][1] == len(first_data_output.shape)
        if len(first_data_output.shape) >= 2:
          assert output_dim[target][0] == first_data_output.shape[1]
      else:
        assert len(first_data_output.shape) == 2, "We expect not sparse. Or specify it explicitly in output_dim."
        output_dim[target] = [first_data_output.shape[1], 2]

    super(StaticDataset, self).__init__(input_dim=input_dim, output_dim=output_dim, num_seqs=num_seqs, **kwargs)

  def generate_seq(self, seq_idx):
    data = self.data[seq_idx]
    return DatasetSeq(seq_idx=seq_idx,
                      features=data["data"],
                      targets={target: data[target] for target in self.target_list})

  def get_target_list(self):
    return self.target_list


class CopyTaskDataset(GeneratingDataset):

  def __init__(self, nsymbols, minlen=0, maxlen=0, minlen_epoch_factor=0, maxlen_epoch_factor=0, **kwargs):
    # Sparse data.
    super(CopyTaskDataset, self).__init__(input_dim=nsymbols,
                                          output_dim={"data": [nsymbols, 1],
                                                      "classes": [nsymbols, 1]},
                                          **kwargs)

    assert nsymbols <= 256
    self.nsymbols = nsymbols
    self.minlen = minlen
    self.maxlen = maxlen
    self.minlen_epoch_factor = minlen_epoch_factor
    self.maxlen_epoch_factor = maxlen_epoch_factor

  def get_random_seq_len(self):
    assert isinstance(self.epoch, int)
    minlen = int(self.minlen + self.minlen_epoch_factor * self.epoch)
    maxlen = int(self.maxlen + self.maxlen_epoch_factor * self.epoch)
    assert 0 < minlen <= maxlen
    return self.random.randint(minlen, maxlen + 1)

  def generate_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    seq_len = self.get_random_seq_len()
    seq = [self.random.randint(0, self.nsymbols) for i in range(seq_len)]
    seq_np = numpy.array(seq, dtype="int8")
    return DatasetSeq(seq_idx=seq_idx, features=seq_np, targets={"classes": seq_np})


class _TFKerasDataset(CachedDataset2):
  """
  Wraps around any dataset from tf.contrib.keras.datasets.
  See: https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/keras/datasets
  TODO: Should maybe be moved to a separate file. (Only here because of tf.contrib.keras.datasets.reuters).
  """
  # TODO...


class _NltkCorpusReaderDataset(CachedDataset2):
  """
  Wraps around any dataset from nltk.corpus.
  TODO: Should maybe be moved to a separate file, e.g. CorpusReaderDataset.py or so?
  """
  # TODO ...


class NltkTimitDataset(CachedDataset2):
  """
  DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus

  This Dataset will get TIMIT via NLTK.
  Demo:

      tools/dump-dataset.py "{'class': 'NltkTimitDataset'}"

  Note: The NLTK data only contains the train data.
  Not sure how useful this is...
  """

  FeatureDim = 13
  # via: https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
  PhoneMapTo39 = {
    'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'ay': 'ay', 'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh',
    'el': 'l', 'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g',
    'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh',
    'k': 'k', 'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow', 'oy': 'oy',
    'p': 'p', 'pau': 'sil', 'pcl': 'sil', 'q': 'q', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil',
    'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh'}
  PhoneMapTo48 = {
    'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ax': 'ax', 'ax-h': 'ax', 'axr': 'er',
    'ay': 'ay', 'b': 'b', 'bcl': 'vcl', 'ch': 'ch', 'd': 'd', 'dcl': 'vcl', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh',
    'el': 'el', 'em': 'm', 'en': 'en', 'eng': 'ng', 'epi': 'epi', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g',
    'gcl': 'vcl', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ix', 'iy': 'iy', 'jh': 'jh',
    'k': 'k', 'kcl': 'cl', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow', 'oy': 'oy',
    'p': 'p', 'pau': 'sil', 'pcl': 'cl', 'q': 'q', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'cl',
    'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'zh'}

  def __init__(self, nltk_download_dir=None, with_delta=False, train=True, **kwargs):
    super(NltkTimitDataset, self).__init__(**kwargs)
    if with_delta:
      self.num_inputs = self.FeatureDim * 2
    else:
      self.num_inputs = self.FeatureDim
    self.labels = sorted(set(self.PhoneMapTo48.values()))
    # Make 'sil' the 0 phoneme.
    self.labels.remove("sil")
    self.labels.insert(0, "sil")
    self.num_outputs = {"data": (self.num_inputs, 2), "classes": (len(self.labels), 1)}

    import os
    try:
      import nltk
    except ImportError:
      print("pip3 install --user nltk")
      raise
    # Alternatives: python_speech_features, talkbox.features.mfcc, librosa
    try:
      import librosa
    except ImportError:
      print("pip3 install --user librosa")
      raise

    from nltk.downloader import Downloader
    downloader = Downloader(download_dir=nltk_download_dir)
    print("NLTK corpus download dir:", downloader.download_dir, file=log.v3)
    timit_dir = downloader.download_dir + "/corpora/timit"
    if not os.path.exists(timit_dir):
      assert downloader.download("timit")
      assert os.path.exists(timit_dir)
    assert os.path.exists(timit_dir + "/timitdic.txt"), "TIMIT download broken? remove the directory %r" % timit_dir

    from nltk.data import FileSystemPathPointer
    from nltk.corpus.reader.timit import TimitCorpusReader, SpeakerInfo
    data_reader = TimitCorpusReader(FileSystemPathPointer(timit_dir))
    utterance_ids = data_reader.utteranceids()
    assert isinstance(utterance_ids, list)
    assert utterance_ids

    split = int(len(utterance_ids) * 0.9)
    if train:
      utterance_ids = utterance_ids[:split]
    else:
      utterance_ids = utterance_ids[split:]

    self._data = []  # list of (seq_tag, mfccs, phone_id_seq)

    for seq_tag in utterance_ids:
      phone_seq = data_reader.phones(seq_tag)
      phone_id_seq = numpy.array([self.labels.index(self.PhoneMapTo48[p]) for p in phone_seq], dtype="int32")
      # word_seq = data_reader.words(utter)
      spk = data_reader.spkrid(seq_tag)
      info = data_reader.spkrinfo(spk)
      assert isinstance(info, SpeakerInfo)
      # see: https://github.com/rdadolf/fathom/blob/master/fathom/speech/preproc.py
      # and: https://groups.google.com/forum/#!topic/librosa/V4Z1HpTKn8Q
      audio, sample_rate = librosa.load("%s/%s.wav" % (timit_dir, seq_tag), sr=None)
      window_len = 0.025
      step_len = 0.010
      mfccs = librosa.feature.mfcc(
        audio, sr=sample_rate,
        n_mfcc=self.FeatureDim,
        hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
      energy = librosa.feature.rmse(
        audio,
        hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
      mfccs[0] = energy  # replace first MFCC with energy, per convention
      assert mfccs.shape[0] == self.FeatureDim
      if with_delta:
        deltas = librosa.feature.delta(mfccs)
        mfccs = numpy.vstack([mfccs, deltas])
      mfccs = mfccs.transpose().astype("float32")
      self._data.append((seq_tag, mfccs, phone_id_seq))

    self._num_seqs = len(self._data)
    self._seq_order = list(range(self._num_seqs))

  def init_seq_order(self, epoch=None, seq_list=None):
    assert seq_list is None
    super(NltkTimitDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    self._num_seqs = len(self._data)
    self._seq_order = self.get_seq_order_for_epoch(
      epoch=epoch, num_seqs=self._num_seqs, get_seq_len=lambda i: len(self._data[i][1]))
    return True

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    if seq_idx >= len(self._seq_order):
      return None
    seq_tag, mfccs, phone_id_seq = self._data[self._seq_order[seq_idx]]
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=mfccs, targets=phone_id_seq)


def demo():
  import better_exchook
  better_exchook.install()
  log.initialize(verbosity=[5])
  import sys
  dsclazzeval = sys.argv[1]
  dataset = eval(dsclazzeval)
  assert isinstance(dataset, Dataset)
  assert isinstance(dataset, GeneratingDataset), "use tools/dump-dataset.py for a generic demo instead"
  assert dataset._input_classes and dataset._output_classes
  assert dataset.num_outputs["data"][1] == 2  # expect 1-hot
  assert dataset.num_outputs["classes"][1] == 1  # expect sparse
  for i in range(10):
    print("Seq idx %i:" % i)
    s = dataset.generate_seq(i)
    assert isinstance(s, DatasetSeq)
    features = s.features
    output_seq = s.targets["classes"]
    assert features.ndim == 2
    assert output_seq.ndim == 1
    input_seq = numpy.argmax(features, axis=1)
    input_seq_str = "".join([dataset._input_classes[i] for i in input_seq])
    output_seq_str = "".join([dataset._output_classes[i] for i in output_seq])
    print(" %r" % input_seq_str)
    print(" %r" % output_seq_str)
    assert features.shape[1] == dataset.num_outputs["data"][0]
    assert features.shape[0] == output_seq.shape[0]


if __name__ == "__main__":
  demo()
