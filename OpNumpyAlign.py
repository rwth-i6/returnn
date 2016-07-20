import numpy as np

import theano

class NumpyAlignOp(theano.Op):
  # Properties attribute
  __props__ = ()

  # index_in, index_out, scores, transcription
  itypes = [theano.tensor.bmatrix,theano.tensor.bmatrix,theano.tensor.ftensor3,theano.tensor.imatrix]
  otypes = [theano.tensor.imatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, index_out, scores, transcriptions = inputs_storage[:4]
    alignment = np.zeros(index_in.shape,'int32')
    for b in xrange(scores.shape[1]):
      length_x = index_in[:,b].sum()
      length_y = index_out[:,b].sum()
      alignment[:length_x,b] = self._fullAlignmentSequence(0, length_x, scores[:length_x,b], transcriptions[:length_y,b])
    output_storage[0][0] = alignment

  # optional:
  #check_input = True

  def __init__(self, *args): # TODO
    self.numStates = 3
    self.repetitions = 1
    self.silence = False
    self.pruningThreshold = 500.
    pass

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
        for r in xrange(self.repetitions):
          hmm[self.silence + i * self.numStates * self.repetitions + s * self.repetitions + r] = \
            self.repetitions * (startState + s) - 1

    return hmm

  def _fullAlignmentSequence(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end with given transcription"""
    # TODO
    tdp = [ 3., 0., 5., 0.]
    stdp = [0., 3., 1000000.0, 0.]

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
          rightScore[s] = leftScore[s] + stdp[0]
          bt[t][s] = 0

        # s is 1th state -> one after silence
        elif s == 1:
          if leftScore[s] + tdp[0] < leftScore[s - 1] + stdp[1]:
            # 0 transition
            rightScore[s] = leftScore[s] + tdp[0]
            bt[t][s] = 0
          else:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + stdp[1]
            bt[t][s] = 1

        # s is 2th state -> two after silence
        elif s == 2:
          if leftScore[s] + tdp[0] < leftScore[s - 1] + tdp[1] and \
                leftScore[s] + tdp[0] < \
                leftScore[s - 2] + stdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + tdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + tdp[1] < \
              leftScore[s - 2] + stdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + stdp[2]
            bt[t][s] = 2
        # s is last state -> silence
        elif s == lengthS - 1:
          if leftScore[s] + stdp[0] < leftScore[s - 1] + tdp[1] and \
                leftScore[s] + stdp[0] < leftScore[s - 2] + tdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + stdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + tdp[1] < leftScore[s - 2] + tdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + tdp[2]
            bt[t][s] = 2
        # s is another state
        else:
          if leftScore[s] + tdp[0] < leftScore[s - 1] + tdp[1] and \
                leftScore[s] + tdp[0] < leftScore[s - 2] + tdp[2]:
            # 0 transition
            rightScore[s] = leftScore[s] + tdp[0]
            bt[t][s] = 0
          elif leftScore[s - 1] + tdp[1] < leftScore[s - 2] + tdp[2]:
            # 1 transition
            rightScore[s] = leftScore[s - 1] + tdp[1]
            bt[t][s] = 1
          else:
            # 2 transition
            rightScore[s] = leftScore[s - 2] + tdp[2]
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

numpyAlignOp = NumpyAlignOp()
