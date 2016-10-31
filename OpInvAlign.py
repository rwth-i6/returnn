import numpy as np

import theano

class InvAlignOp(theano.Op):
  # Properties attribute
  __props__ = ('tdps',)

  # index_in, index_out, scores, transcription
  itypes = [theano.tensor.bmatrix,theano.tensor.bmatrix,theano.tensor.ftensor3,theano.tensor.imatrix]
  otypes = [theano.tensor.imatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, index_out, scores, transcriptions = inputs_storage[:4]
    alignment = np.zeros(index_in.shape,'int32')
    for b in range(scores.shape[1]):
      length_x = index_in[:,b].sum()
      length_y = index_out[:,b].sum()
      alignment[:length_x, b] = self._fullSegmentationInv(0, length_x, scores[:length_x, b],
                                                          transcriptions[:length_y, b])
    output_storage[0][0] = alignment

  def __init__(self, tdps): # TODO
    self.numStates = 1
    self.pruningThreshold = 500.
    self.tdps = tdps

  def grad(self, inputs, output_grads):
    return [output_grads[0] * 0]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[0]]

  def _buildHmm(self, transcription):
    """Builds list of hmm states (with repetitions) for transcription"""
    transcriptionLength = transcription.shape[0]
    hmm = np.zeros(transcriptionLength * self.numStates, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.numStates + 1
      for s in range(0, self.numStates):
        hmm[i * self.numStates + s] = startState + s - 1

    return hmm


  def _fullAlignmentSequenceInv(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    # max skip transitions derived from tdps
    skip = len(self.tdps)

    hmm = self._buildHmm(transcription)
    lengthT = end - start
    lengthS = transcription.shape[0] * self.numStates * self.repetitions

    leftScore = np.full((lengthT + skip - 1), inf, dtype=np.float32)
    rightScore = np.full((lengthT + skip - 1), inf, dtype=np.float32)
    bt = np.zeros((lengthS, lengthT), dtype=np.int)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    densities = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.numStates]

    # initialize first column
    leftScore[0 + skip - 1] = score[0][0 + skip - 1]

    # go through all following columns except last (silence)
    for s in range(1, lengthS):
      for t in range(0, lengthT):
        # scores calculates just as in recognition
        scores = np.add(np.add(
            np.cumsum(np.append(
                [0.], score[s, t + 1:t + skip][::-1])), self.tdps),
                leftScore[t:t + skip][::-1])

        # index corresponds to transition
        bestChoice = np.argmin(scores)

        rightScore[t + skip - 1] = scores[bestChoice]
        bt[s][t] = bestChoice

      leftScore, rightScore = rightScore, leftScore
      rightScore = np.full((lengthT + skip - 1), inf, dtype=np.float32)

    # backtrack alignment
    result = [0] * lengthT
    t = lengthT - 1
    for s in range(lengthS - 1, -1, -1):
      for span in range(0, bt[s][t]):
          result[start + t - span] = int((hmm[s]/self.numStates + 1))
      t = t - bt[s][t]
      assert t >= 0, "invalid alignment"

    # handle remaining timeframes -> silence
    for span in range(0, t + 1):
      result[start + span] = int((hmm[s]/self.numStates + 1))
    return result


  def _fullSegmentationInv(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    # max skip transitions derived from tdps
    skip = len(self.tdps)

    hmm = self._buildHmm(transcription)
    lengthT = end - start
    lengthS = transcription.shape[0] * self.numStates * self.repetitions

    # with margins of skip at the bottom or top
    fwdScore = np.full((lengthS, lengthT + skip - 1), inf)
    bt = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int32)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.numStates]

      # forward
      # initialize first column
      # assume virtual start at t = -1
      scores = score[0, 0 + skip - 1:skip + skip - 2]
      #if allFeatures:
      #  scores = np.cumsum(scores)
      scores = np.add(scores, self.tdps[1:])
      fwdScore[0, 0 + skip - 1:skip + skip - 2] = scores
      bt[0, 0 + skip - 1:skip + skip - 2] = range(1, skip)

      # remaining columns
      for s in range(1, lengthS):
        for t in range(0, lengthT):
          previous = fwdScore[s - 1, t:t + skip]
          scores = score[s, t + skip - 1]
          scores = np.add(scores, np.add(previous, self.tdps[::-1]))

          best = np.argmin(scores)
          fwdScore[s, t + skip - 1] = scores[best]
          bt[s, t + skip - 1] = skip - 1 - best

      alignment = np.full((lengthT), -2, dtype=np.int32)
      # backtrack
      t = lengthT - 1
      alignment[t] = hmm[lengthS - 1]
      for s in range(lengthS - 2, -1, -1):
        tnew = t - bt[s + 1][t + skip - 1]
        alignment[tnew:t] = hmm[s]
        t = tnew

      alignment[0:t] = 0

      assert not -2 in alignment
      return alignment



invAlignOp = InvAlignOp([ 1e10, 0., 1.9, 3., 2.5, 2., 1.4 ])
