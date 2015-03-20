

class Batch:
  """
  A batch can consists of several sequences (= segments).
  Note that self.shape[1] is a different kind of batch - related to the data-batch-idx (= seq-idx).
  """

  def __init__(self, start = (0, 0)):
    """
    :type start: list[int]
    """
    self.shape = [0, 0]  # format (time,batch)
    self.start = list(start)  # format (start seq idx in data, start frame idx in seq)
    self.nseqs = 1
    """
    nseqs is the number of sequences which we cover (not data-batches self.shape[1]).
    For recurrent NN training, shape[1] == nseqs.
    For FF NN training, we concatenate all seqs, so shape[1] == 1 but nseqs >= 1.
    """

  def __repr__(self):
    return "<Batch %r %r>" % (self.shape, self.start)

  def try_sequence(self, length):
    """
    :param int length: number of (time) frames
    :return: new shape which covers the old shape and one more data-batch
    :rtype: list[int]
    """
    return [max(self.shape[0], length), self.shape[1] + 1]

  def add_sequence(self, length):
    """
    Adds one data-batch.
    :param int length: number of (time) frames
    """
    self.shape = self.try_sequence(length)

  def add_frames(self, length):
    """
    Adds frames to all data-batches.
    Will add one data-batch if we don't have one yet.
    :param int length: number of (time) frames
    """
    self.shape = [self.shape[0] + length, max(self.shape[1], 1)]

  def size(self):
    return self.shape[0] * self.shape[1]

  def get_end_seq(self):
    return self.start[0] + max(self.nseqs, self.shape[1])
