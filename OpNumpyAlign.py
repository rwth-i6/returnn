import numpy as np

import theano

class NumpyAlignOp(theano.Op):
  # Properties attribute
  __props__ = ('inverse',)

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
      if self.inverse:
        alignment[:length_x,b] = self._fullAlignmentSequenceInv(0, length_x, scores[:length_x,b],
                                                             transcriptions[:length_y,b])
      else:
        #alignment[:length_x,b] = self._fullAlignmentSequence(0, length_x, scores[:length_x,b],
        #                                                        transcriptions[:length_y,b])
        alignment[:length_x,b] = self._ViterbiSequence(0, length_x, scores[:length_x,b],transcriptions[:length_y,b])
    output_storage[0][0] = alignment

  # optional:
  #check_input = True

  def __init__(self, inverse): # TODO
    self.numStates = 3
    self.inverse = inverse
    self.repetitions = 1
    self.silence = True
    self.pruningThreshold = 500.
    if inverse:
      self.tdp = [ 1e10, 0., 1.9, 3., 2.5, 2., 1.4 ]
    else:
      self.tdp = [ 3., 0., 5., 0.] # loop forward skip exit(?)
      self.stdp = [0., 3., 1000000.0, 0.]

  def grad(self, inputs, output_grads):
    return [output_grads[0] * 0]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[0]]

  def _buildHmm(self, transcription):
    """Builds list of hmm states (with repetitions) for transcription"""
    transcriptionLength = transcription.shape[0]

    if self.silence:
      hmm = np.zeros(self.repetitions + transcriptionLength * self.numStates * self.repetitions, dtype=np.int32)
    else:
      hmm = np.zeros(transcriptionLength * self.numStates * self.repetitions, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.numStates + 1
      for s in range(0, self.numStates):
        for r in range(self.repetitions):
          hmm[self.silence + i * self.numStates * self.repetitions + s * self.repetitions + r] = \
            self.repetitions * (startState + s) - 1

    return hmm

  def _ViterbiSequence(self, start, end, scores, transcription):
    """Align a given sequence with the full sum"""

    usedStates = self.numStates * 2
    inf = 1e30
    hmm = self._buildHmm(transcription)
    lengthT = end - start
    lengthS = len(hmm)

    # with margins of 2 at the bottom or top
    fwdScore = np.full((lengthT, lengthS + 2), inf)
    bt = np.full((lengthT, lengthS + 2), -1, dtype=np.int32)

    # precompute all state priors
    stateprior = np.full((lengthS), inf)
    for s in range(0, lengthS):
      stateprior[s] = 0.0 #self.stateprior((hmm[s] + 1)/self.repetitions) * prior_scale

    # precompute all scores and densities
    score = np.full((lengthT, lengthS), inf)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        h = (hmm[s] + 1) / self.repetitions
        h -= h % 3
        h -= 1
        h /= self.numStates
        score[t][s] = scores[start + t, h]
    # divide all scores with state priors
    #score = np.subtract(score, stateprior)

    # forward
    # initialize first column
    fwdScore[0, 2] = score[0, 0]

    # go through all following columns
    for t in range(1, lengthT):
      for s in range(0, lengthS):
        scores = np.add(np.add(fwdScore[t - 1, s:s + 3], score[t][s]), self.tdp[0:3][::-1])
        best = np.argmin(scores)
        fwdScore[t][s + 2] = scores[best]
        bt[t][s + 2] = 2 - best

    alignment = np.full((lengthT), -1, dtype=np.int32)
    # backtrack
    s = lengthS - 1
    alignment[lengthT - 1] = hmm[s]
    for t in range(lengthT - 2, -1, -1):
      s = s - bt[t + 1][s + 2]
      alignment[t] = hmm[s]

    return alignment

  def _fullAlignmentSequence(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end with given transcription"""
    inf = 1e30
    hmm = self._buildHmm(transcription)

    lengthT = end - start
    if self.silence:
      lengthS = self.repetitions + transcription.shape[0] * self.numStates * self.repetitions
    else:
      lengthS = transcription.shape[0] * self.numStates * self.repetitions

    # leftScore = np.array([inf] * lengthS)
    leftScore = np.full((lengthS,), inf, dtype=np.float32)
    # rightScore = np.array([inf] * lengthS)
    rightScore = np.full((lengthS,), inf, dtype=np.float32)
    bt = np.zeros((lengthT, lengthS), dtype=np.int32)

    # initialize first column
    leftScore[0] = scores[start,transcription[0]]
    bestLeftScore = leftScore[0]
    bestRightScore = inf
    bt[0][0] = 0

    # go through all following columns
    for t in range(1, lengthT):
      for s in range(0, lengthS):

        # s is 0th state -> silence
        if s == 0:
          # 0 transition
          rightScore[s] = leftScore[s] + self.stdp[0]
          bt[t][s] = 0

        # s is 1th state -> one after silence
        elif s == 1:
          if leftScore[s] + self.tdp[0] < leftScore[s - 1] + self.stdp[1]:
            # 0 transition
            rightScore[s] = leftScore[s] + self.tdp[0]
            bt[t][s] = 0
          else:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + self.stdp[1]
            bt[t][s] = 1

        # s is 2th state -> two after silence
        elif s == 2:
          if leftScore[s] + self.tdp[0] < leftScore[s - 1] + self.tdp[1] and \
                leftScore[s] + self.tdp[0] < \
                leftScore[s - 2] + self.stdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + self.tdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + self.tdp[1] < \
              leftScore[s - 2] + self.stdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + self.tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + self.stdp[2]
            bt[t][s] = 2
        # s is last state -> silence
        elif s == lengthS - 1:
          if leftScore[s] + self.stdp[0] < leftScore[s - 1] + self.tdp[1] and \
                leftScore[s] + self.stdp[0] < leftScore[s - 2] + self.tdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + self.stdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + self.tdp[1] < leftScore[s - 2] + self.tdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + self.tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + self.tdp[2]
            bt[t][s] = 2
        # s is another state
        else:
          if leftScore[s] + self.tdp[0] < leftScore[s - 1] + self.tdp[1] and \
                leftScore[s] + self.tdp[0] < leftScore[s - 2] + self.tdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + self.tdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + self.tdp[1] < leftScore[s - 2] + self.tdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + self.tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + self.tdp[2]
            bt[t][s] = 2

        # Pruning
        # if bestRightScore > rightScore[s]:
        #     bestRightScore = rightScore[s]
        # if rightScore[s] < bestLeftScore + pruningThreshold:
        tempScore = scores[start + t, hmm[s] / self.numStates ] # we consider the same emission for all states
        #tempScore, dens[t][s] = scores[start + t, transcription[s]] # TODO: this works only for single state/repetition
        rightScore[s] += tempScore

      leftScore, rightScore = rightScore, leftScore
      bestLeftScore = bestRightScore
      # rightScore = np.array([inf] * lengthS)
      rightScore = np.full((lengthS), inf, dtype=np.float64)
      bestRightScore = inf
    result = np.zeros((lengthT,),'int32')
    # backtrack alignment
    s = lengthS - 1
    for t in range(lengthT - 1, -1, -1):
      result[start + t] = int((hmm[s] / self.numStates + 1) / self.repetitions)
      s = s - bt[t][s]
      assert s >= 0, "invalid alignment"
    #print result
    return result

  def _fullAlignmentSequenceInv(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    # max skip transitions derived from tdps
    skip = len(self.tdp)

    hmm = self._buildHmm(transcription)
    lengthT = end - start
    if self.silence:
      lengthS = self.repetitions + transcription.shape[0] * self.numStates * self.repetitions
    else:
      lengthS = transcription.shape[0] * self.numStates * self.repetitions

    leftScore = np.full((lengthT + skip - 1), inf, dtype=np.float64)
    rightScore = np.full((lengthT + skip - 1), inf, dtype=np.float64)
    bt = np.zeros((lengthS, lengthT), dtype=np.int)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    densities = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.numStates]

    # initialize first column
    if self.silence:
      leftScore[0 + skip - 1:lengthT + skip - 1] = \
          np.cumsum(score[0, 0 + skip - 1:lengthT + skip - 1])
    else:
      # no silence at the beginning
      leftScore[0 + skip - 1] = score[0][0 + skip - 1]

    bestLeftScore = leftScore[0 + skip - 1]
    bestRightScore = inf

    # go through all following columns except last (silence)
    for s in range(1, lengthS - self.silence):
      for t in range(0, lengthT):
        # scores calculates just as in recognition
        scores = np.add(np.add(
            np.cumsum(np.append(
                [0.], score[s, t + 1:t + skip][::-1])), self.tdp),
                leftScore[t:t + skip][::-1])

        # index corresponds to transition
        bestChoice = np.argmin(scores)

        rightScore[t + skip - 1] = scores[bestChoice]
        bt[s][t] = bestChoice

      leftScore, rightScore = rightScore, leftScore
      bestLeftScore = bestRightScore
      rightScore = np.full((lengthT + skip - 1), inf, dtype=np.float64)
      bestRightScore = inf

    # handle last column (silence) with 1 transitions
    if self.silence:
      s = lengthS - 1
      for t in range(1, lengthT):
        if leftScore[t + skip - 1] > leftScore[t + skip - 2] + score[s][t + skip - 1]:
          # do 1 transition in silence
          leftScore[t + skip - 1] = \
              leftScore[t + skip - 2] + score[s][t + skip - 1]
          bt[s][t] = bt[s][t - 1] + 1

    # backtrack alignment
    result = [0] * lengthT
    t = lengthT - 1
    for s in range(lengthS - 1, -1, -1):
      for span in range(0, bt[s][t]):
          result[start + t - span] = int((hmm[s]/self.numStates + 1) / self.repetitions)
      t = t - bt[s][t]
      assert t >= 0, "invalid alignment"

    # handle remaining timeframes -> silence
    for span in range(0, t + 1):
      result[start + span] = int((hmm[s]/self.numStates + 1) / self.repetitions)
    return result

numpyAlignOp = NumpyAlignOp(False)
