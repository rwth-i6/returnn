import numpy as np

import theano

class InvAlignOp(theano.Op):
  __props__ = ('tdps','nstates')

  # index_in, index_out, scores, transcriptions
  itypes = [theano.tensor.bmatrix,theano.tensor.bmatrix,theano.tensor.ftensor3,theano.tensor.imatrix]
  otypes = [theano.tensor.imatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, index_out, scores, transcriptions = inputs_storage[:4]
    attention = np.zeros(index_out.shape, 'int32')
    for b in range(scores.shape[1]):
      length_x = index_in[:,b].sum()
      length_y = index_out[:,b].sum()
      attention[:length_y, b] = self._viterbi(0, length_x, scores[:length_x, b],
                                              transcriptions[:length_y / self.nstates, b])
      attention[:,b] += b * index_in.shape[0]
    output_storage[0][0] = attention

  def __init__(self, tdps, nstates):
    self.nstates = nstates
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0] * 0]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[1]]

  def _buildHmm(self, transcription):
    """Builds list of hmm states for transcription"""
    transcriptionLength = transcription.shape[0]
    hmm = np.zeros(transcriptionLength * self.nstates, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.nstates + 1
      for s in range(0, self.nstates):
        hmm[i * self.nstates + s] = startState + s - 1

    return hmm

  def _viterbi(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    # max skip transitions derived from tdps

    hmm = self._buildHmm(transcription)
    lengthT = end - start
    skip = min(len(self.tdps), lengthT - self.nstates)
    tdps = self.tdps[:skip]
    lengthS = transcription.shape[0] * self.nstates

    # with margins of skip at the bottom or top
    fwdScore = np.full((lengthS, lengthT + skip - 1), inf)
    bt = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int32)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.nstates]

    # forward
    # initialize first column
    # assume virtual start at t = -1
    scores = score[0, 0 + skip - 1:skip + skip - 2]
    # if allFeatures:
    #  scores = np.cumsum(scores)
    scores = np.add(scores, tdps[1:])
    fwdScore[0, 0 + skip - 1:skip + skip - 2] = scores
    bt[0, 0 + skip - 1:skip + skip - 2] = range(1, skip)

    # remaining columns
    for s in range(1, lengthS):
      for t in range(0, lengthT):
        previous = fwdScore[s - 1, t:t + skip]
        scores = score[s, t + skip - 1]
        scores = np.add(scores, np.add(previous, tdps[::-1]))

        best = np.argmin(scores)
        fwdScore[s, t + skip - 1] = scores[best]
        bt[s, t + skip - 1] = skip - 1 - best

    attention = np.full((lengthS), -1, dtype=np.int32)

    # backtrack
    t = lengthT - 1
    attention[lengthS - 1] = lengthT - 1
    for s in range(lengthS - 2, -1, -1):
      tnew = t - bt[s + 1][t + skip - 1]
      attention[s] = tnew
      t = tnew
    return attention


class InvDecodeOp(theano.Op):
  # Properties attribute
  __props__ = ('tdps', 'nstates')

  # index_in, scores
  itypes = [theano.tensor.bmatrix, theano.tensor.ftensor3]
  otypes = [theano.tensor.imatrix, theano.tensor.imatrix, theano.tensor.bmatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, scores = inputs_storage[:2]
    transcript = np.zeros(index_in.shape,'int32')
    attention = np.zeros(index_in.shape, 'int32')
    index = np.zeros(index_in.shape, 'int8')
    max_length_y = 0
    for b in range(scores.shape[1]):
      length_x = index_in[:,b].sum()
      t, a = self._recognize(0, length_x, scores[:length_x, b], 3.0)
      length_y = len(a)
      transcript[:length_y, b] = t
      attention[:length_y, b] = a + b * index_in.shape[0]
      index[:length_y, b] = 1
      max_length_y = max(length_y, max_length_y)

    output_storage[0][0] = transcript[:max_length_y]
    output_storage[1][0] = attention[:max_length_y]
    output_storage[2][0] = index[:max_length_y]

  def __init__(self, tdps, nstates): # TODO
    self.nstates = nstates
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0] * 0, output_grads[1] * 0, output_grads[1] * 0]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[0], input_shapes[0], input_shapes[0]]

  def _buildHmm(self, transcription):
    """Builds list of hmm states (with repetitions) for transcription"""
    transcriptionLength = transcription.shape[0]
    hmm = np.zeros(transcriptionLength * self.nstates, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.nstates + 1
      for s in range(0, self.nstates):
        hmm[i * self.nstates + s] = startState + s - 1

    return hmm

  def _recognize(self, start, end, scores, wordPenalty):
    """recognizes a sequence from start until (excluded) end
    with inverse search s -> t_s"""
    inf = 1e30
    bestAttend = []
    bestResult = []
    bestScore = inf

    skip = len(self.tdps)
    maxDuration = 0.8
    mod_factor = 1.0

    hmmLength = scores.shape[1] * self.nstates
    seqLength = end - start
    leftScore = np.full((hmmLength, seqLength + skip - 1), inf)
    rightScore = np.full((hmmLength, seqLength + skip - 1), inf)

    leftStart = np.full((hmmLength, seqLength + skip - 1), 0,
                        dtype=np.uint64)
    rightStart = np.full((hmmLength, seqLength + skip - 1), 0,
                         dtype=np.uint64)

    leftEpoch = np.full((hmmLength, seqLength + skip - 1), - 1,
                        dtype=np.int64)
    rightEpoch = np.full((hmmLength, seqLength + skip - 1), - 1,
                         dtype=np.int64)

    bestWordEnds = \
      np.full((int(maxDuration * seqLength), seqLength), - 1,
              dtype=np.int64)
    bestWordEndsStart = \
      np.full((int(maxDuration * seqLength), seqLength), - 1,
              dtype=np.int64)
    bestWordEndsEpoch = \
      np.full((int(maxDuration * seqLength), seqLength), - 1,
              dtype=np.int64)

    # precompute all scores and densities
    score = np.full((hmmLength, seqLength + skip), inf)
    for t in range(0, seqLength):
      for s in range(0, hmmLength):
        score[s][t + skip - 1] = scores[start + t, s / self.nstates]

    # apply model scale
    score = np.multiply(score, mod_factor)

    # initalize for first timeframe
    leftScore[0][skip - 1] = score[0][skip - 1]
    leftEpoch[0][skip - 1] = 0
    bestWordEnds[0][0] = 0
    bestWordEndsStart[0][0] = 0
    bestWordEndsEpoch[0][0] = 0

    for s in range(1, hmmLength, self.nstates):
      leftScore[s][skip - 1] = score[s][skip - 1]
      leftEpoch[s][skip - 1] = 0

    for a in range(1, int(maxDuration * seqLength)):
      # determine best word ends and score
      bestWordEnd = np.argmin(np.concatenate(
        ([leftScore[0]],
         leftScore[self.nstates:hmmLength:
         self.nstates] + wordPenalty)), axis=0)

      bestWordEnd = bestWordEnd * self.nstates

      # two times min might be improved
      bestWordEndScore = np.min(np.concatenate(
        ([leftScore[0]],
         leftScore[self.nstates:hmmLength:
         self.nstates] + wordPenalty)), axis=0)

      bestWordEnds[a] = bestWordEnd[skip - 1::]
      for t in range(0, seqLength):
        bestWordEndsStart[a][t] = \
          leftStart[bestWordEnd[t + skip - 1]][t + skip - 1]
        bestWordEndsEpoch[a][t] = \
          leftEpoch[bestWordEnd[t + skip - 1]][t + skip - 1]

      for s in range(0, hmmLength):
        # for silence allow special transitions (only 1)
        if s == 0:
          for t in range(0, seqLength - 1):
            if leftScore[0][t + skip - 1] < \
              bestWordEndScore[t + skip - 1] or \
                bestWordEnd[t + skip - 1] == 0:
              # inner silence
              rightScore[0][t + skip] = \
                leftScore[0][t + skip - 1] + score[0][t + skip]
              rightStart[0][t + skip] = \
                leftStart[0][t + skip - 1]
              rightEpoch[0][t + skip] = \
                leftEpoch[0][t + skip - 1]
            else:
              # to silence
              # only transition 1 is allowed
              rightScore[0][t + skip] = \
                bestWordEndScore[t + skip - 1] \
                + score[0][t + skip]
              rightStart[0][t + skip] = t + 1
              rightEpoch[0][t + skip] = a

        # between word transitions
        elif s % self.nstates == 1:
          for t in range(0, seqLength):
            # linear interpolate scores
            scores = np.add(np.add(
              np.cumsum(np.append(
                [0.], score[s, t + 1:t + skip][::-1])), self.tdps),
              bestWordEndScore[t:t + skip][::-1])

            # index corresponds to transition
            bestChoice = np.argmin(scores)

            rightScore[s][t + skip - 1] = scores[bestChoice]
            rightStart[s][t + skip - 1] = t + 1 - bestChoice
            rightEpoch[s][t + skip - 1] = a

        # inner word transitions
        else:
          for t in range(0, seqLength):
            # linear interpolate scores
            scores = np.add(np.add(
              np.cumsum(np.append(
                [0.], score[s, t + 1:t + skip][::-1])), self.tdps),
              leftScore[s - 1, t:t + skip][::-1])

            # index corresponds to transition
            bestChoice = np.argmin(scores)

            rightScore[s][t + skip - 1] = \
              scores[bestChoice]
            rightStart[s][t + skip - 1] = \
              leftStart[s - 1][t + skip - 1 - bestChoice]
            rightEpoch[s][t + skip - 1] = \
              leftEpoch[s - 1][t + skip - 1 - bestChoice]

      # finish epoch
      leftScore, rightScore = rightScore, leftScore
      leftStart, rightStart = rightStart, leftStart
      leftEpoch, rightEpoch = rightEpoch, leftEpoch

      rightScore = np.full((hmmLength, seqLength + skip - 1), inf)
      rightStart = np.full((hmmLength, seqLength + skip - 1), start,
                           dtype=np.uint64)
      rightEpoch = np.full((hmmLength, seqLength + skip - 1), - 1,
                           dtype=np.int64)

      # backtrack
      # backtrace = []
      if bestWordEndScore[-1] < inf:
        result = []
        attend = []
        t_idx = seqLength - 1
        a_idx = a
        while t_idx > 0:
          result.append(a_idx / self.nstates)
          attend.append(t_idx)

          # backtrace.append((result[-1],
          #     t_idx, bestWordEndsStart[a_idx][t_idx] - 1,
          #     a_idx, bestWordEndsEpoch[a_idx][t_idx]))
          t_idx_temp = bestWordEndsStart[a_idx][t_idx] - 1
          a_idx = bestWordEndsEpoch[a_idx][t_idx]
          t_idx = t_idx_temp

        if bestWordEndScore[-1] < bestScore:
          bestResult = list(reversed(result))
          bestScore = bestWordEndScore[-1]
          bestAttend = list(reversed(attend))
    return bestResult, bestAttend
