

class Batch:
  """
  A batch can consists of several sequences (= segments).
  Note that self.shape[1] is a different kind of batch - related to the data-batch-idx (= seq-idx).
  """

  def __init__(self, start=(0, 0)):
    """
    :type start: list[int]
    """
    self.data_shape = [0, 0]  # format (time,batch)
    self.start = list(start)  # format (start seq idx in data, start frame idx in seq)
    self.nseqs = 1
    """
    nseqs is the number of sequences which we cover (not data-batches self.shape[1]).
    For recurrent NN training, data_shape[1] == nseqs,
    and we ignore nseqs.
    For FF NN training, we concatenate all seqs, so data_shape[1] == 1 but nseqs >= 1.
    data_shape is the shape of the final data batch given to the device.
    """

  def __repr__(self):
    return "<Batch start:%r data_shape:%r nseqs:%i>" % (self.start, self.data_shape, self.nseqs)

  def try_sequence(self, length):
    """
    :param int length: number of (time) frames
    :return: new shape which covers the old shape and one more data-batch
    :rtype: list[int]
    """
    return [max(self.data_shape[0], length), self.data_shape[1] + 1]

  def add_sequence(self, length):
    """
    Adds one data-batch.
    :param int length: number of (time) frames
    """
    self.data_shape = self.try_sequence(length)

  def add_frames(self, length):
    """
    Adds frames to all data-batches.
    Will add one data-batch if we don't have one yet.
    :param int length: number of (time) frames
    """
    self.data_shape = [self.data_shape[0] + length, max(self.data_shape[1], 1)]

  def get_num_frames(self):
    """
    Note that this is only an upper limit in case of data_shape[1] > 1
    because data_shape[0] is the max frame len of all seqs.
    """
    return self.data_shape[0] * self.data_shape[1]

  @property
  def start_seq(self):
    return self.start[0]

  def get_num_seqs(self):
    """
    In the recurrent case, we ignore nseqs and the number of seqs is data_shape[1].
    In the FF case, we have data_shape[1] and use nseqs.
    """
    return max(self.nseqs, self.data_shape[1])

  def get_end_seq(self):
    return self.start_seq + self.get_num_seqs()


class BatchSetGenerator:
  def __init__(self, dataset, generator):
    """
    :type dataset: Dataset.Dataset
    :type generator: iter[Batch]
    """
    self.dataset = dataset
    self.generator = generator
    self.buffer = []; " :type: list[Batch] "
    self.reached_end = False
    self.last_batch = None; " :type: Batch "
    self.current_batch_idx = 0

  def _read_next(self):
    if self.reached_end:
      return False
    try:
      batch = next(self.generator)
    except StopIteration:
      self.reached_end = True
      return False
    else:
      self.buffer += [batch]
      return True

  def _read_next_up_to_n(self, n):
    for i in range(n):
      if len(self.buffer) >= n:
        break
      if not self._read_next():
        break

  def peek_next_n(self, n):
    """
    :type: list[Batch]
    :returns it might return less. There is no way to know in advance.
    If self.has_more() is True, it will at least return one.
    """
    self._read_next_up_to_n(n)
    return self.buffer[:n]

  def advance(self, n):
    """
    :type n: int
    """
    assert n > 0
    self._read_next_up_to_n(n)
    assert n <= len(self.buffer)
    self.last_batch = self.buffer[n - 1]
    self.buffer = self.buffer[n:]
    self.current_batch_idx += n

  def completed_frac(self):
    """
    :rtype: float
    :returns 0-1
    """
    if not self.last_batch:
      return 0.0
    # We cannot use the batch idx because we don't know the number
    # of batches in advance. Thus, we use the seq idx instead.
    # It's good enough.
    assert self.dataset.num_seqs > 0
    return float(self.last_batch.start_seq) / self.dataset.num_seqs

  def has_more(self):
    """
    :rtype: bool
    """
    if len(self.buffer) > 0:
      return True
    return self._read_next()

  def get_current_batch_idx(self):
    """
    :rtype: int
    """
    return self.current_batch_idx
