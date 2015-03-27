

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


