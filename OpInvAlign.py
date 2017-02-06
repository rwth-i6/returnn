import numpy as np

import theano

class InvAlignOp(theano.Op):
  __props__ = ('tdps','nstates')

  # index_in, index_out, scores, transcriptions
  itypes = [theano.tensor.bmatrix,theano.tensor.bmatrix,theano.tensor.ftensor3,theano.tensor.imatrix]
  otypes = [theano.tensor.imatrix,theano.tensor.imatrix,theano.tensor.bmatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, index_out, scores, transcriptions = inputs_storage[:4]
    shape = (index_out.shape[0] * self.nstates, index_out.shape[1])
    attention = np.zeros(shape, 'int32')
    index = np.zeros(shape, 'int8')
    labelling = np.zeros(shape, 'int32')
    length_x = index_in.sum(axis=0)
    length_y = index_out.sum(axis=0) * self.nstates
    for b in range(scores.shape[1]):
      index[:length_y[b], b] = np.int8(1)
      orth = transcriptions[:length_y[b] / self.nstates, b]
      attention[:length_y[b], b], labelling[:length_y[b], b] = self._viterbi(0, length_x[b], scores[:length_x[b], b], orth)
      attention[:length_y[b], b] += b * index_in.shape[0]
    output_storage[0][0] = labelling
    output_storage[1][0] = attention
    output_storage[2][0] = index
    #print attention

  def __init__(self, tdps, nstates):
    self.nstates = nstates
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0],output_grads[1],output_grads[2],output_grads[1]]

  def infer_shape(self, node, input_shapes):
    shape = (input_shapes[1][0] * self.nstates, input_shapes[1][1])
    return [shape,shape,shape]

  def _buildHmm(self, transcription):
    start = transcription * self.nstates + 1
    hmm = np.expand_dims(start,axis=1).repeat(self.nstates,axis=1) + np.arange(self.nstates) - 1
    return hmm.astype('int32').flatten()

  def _viterbi(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    lengthT = end - start
    skip = max(min(len(self.tdps), lengthT - self.nstates), 1)
    tdps = self.tdps[:skip]
    lengthS = transcription.shape[0] * self.nstates

    hmm = self._buildHmm(transcription)
    # with margins of skip at the bottom or top
    fwdScore = np.full((lengthS, lengthT + skip - 1), inf)
    bt = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int32)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.nstates]

    # forward
    scores = score[0, 0 + skip - 1:skip + skip - 2]
    scores = np.add(scores, tdps[1:])
    fwdScore[0, 0 + skip - 1:skip + skip - 2] = scores
    bt[0, 0 + skip - 1:skip + skip - 2] = range(1, skip)

    # remaining columns
    for s in range(1, lengthS):
      for t in range(max(lengthT - (lengthS - s) * skip,0), lengthT):
        previous = fwdScore[s - 1, t:t + skip]
        scores = score[s, t + skip - 1]
        scores = np.add(scores, np.add(previous, tdps[::-1]))

        best = np.argmin(scores)
        fwdScore[s, t + skip - 1] = scores[best]
        bt[s, t + skip - 1] = skip - 1 - best

    attention = np.full((lengthS), 0, dtype=np.int32)
    labelling = np.full((lengthS), 0, dtype=np.int32)

    # backtrack
    t = lengthT - 1
    attention[lengthS - 1] = lengthT - 1
    labelling[lengthS - 1] = transcription[-1]
    for s in range(lengthS - 2, -1, -1):
      tnew = t - bt[s + 1][t + skip - 1]
      attention[s] = tnew
      labelling[s] = transcription[s / self.nstates]
      t = tnew
    return attention, labelling

  def _viterbi2(self, start, end, scores, transcription):
    """Fully aligns sequence from start to end but in inverse manner"""
    inf = 1e30
    # max skip transitions derived from tdps

    hmm = self._buildHmm(transcription)
    lengthT = end - start
    skip = max(min(len(self.tdps), lengthT - self.nstates),1)
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
    scores = score[0, 0 + skip - 1:skip + skip - 2]
    scores = np.add(scores, tdps[1:])
    fwdScore[0, 0 + skip - 1:skip + skip - 2] = scores
    bt[0, 0 + skip - 1:skip + skip - 2] = range(1, skip)

    # remaining columns
    for s in range(1, lengthS):
      for t in range(max(lengthT - (lengthS - s) * skip,0), lengthT):
        previous = fwdScore[s - 1, t:t + skip]
        scores = score[s, t + skip - 1]
        scores = np.add(scores, np.add(previous, tdps[::-1]))

        best = np.argmin(scores)
        fwdScore[s, t + skip - 1] = scores[best]
        bt[s, t + skip - 1] = skip - 1 - best

    attention = np.full((lengthS), 0, dtype=np.int32)
    labelling = np.full((lengthS), 0, dtype=np.int32)

    # backtrack
    t = lengthT - 1
    attention[lengthS - 1] = lengthT - 1
    labelling[lengthS - 1] = transcription[-1]
    for s in range(lengthS - 2, -1, -1):
      tnew = t - bt[s + 1][t + skip - 1]
      attention[s] = tnew
      labelling[s] = transcription[s / self.nstates]
      t = tnew
    return attention, labelling


class InvFullAlignOp(theano.Op):
  __props__ = ('tdps','nstates')

  # index_in, index_out, scores, transcriptions
  itypes = [theano.tensor.bmatrix,theano.tensor.bmatrix,theano.tensor.ftensor3,theano.tensor.imatrix]
  otypes = [theano.tensor.ftensor3,theano.tensor.bmatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, index_out, scores, transcriptions = inputs_storage[:4]
    index = np.zeros((index_out.shape[0] * self.nstates, index_out.shape[1]), 'int8')
    err = np.zeros(scores.shape, 'float32')

    labelling = np.zeros(shape, 'int32')
    length_x = index_in.sum(axis=0)
    length_y = index_out.sum(axis=0) * self.nstates
    for b in range(scores.shape[1]):
      orth = transcriptions[:length_y[b] / self.nstates, b]
      index[:length_y[b], b] = np.int8(1)
      err[:length_x[b],b] = self._baumwelch(0, length_x[b], scores[:length_x[b], b], orth)
    output_storage[0][0] = err
    output_storage[1][0] = index

  def __init__(self, tdps, nstates):
    self.nstates = nstates
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0],output_grads[1],output_grads[2],output_grads[1]]

  def infer_shape(self, node, input_shapes):
    shape = (input_shapes[1][0] * self.nstates, input_shapes[1][1])
    return [input_shapes[2],(input_shapes[1][0] * self.nstates, input_shapes[1][1])]

  def _buildHmm(self, transcription):
    start = transcription * self.nstates + 1
    hmm = np.expand_dims(start,axis=1).repeat(self.nstates,axis=1) + np.arange(self.nstates) - 1
    return hmm.astype('int32').flatten()

  def _baumwelch(self, start, end, scores, transcription):
    from scipy.misc import logsumexp

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
    bwdScore = np.full((lengthS, lengthT + skip - 1), inf)
    bt = np.full((lengthS, lengthT + skip - 1), -1, dtype=np.int32)

    # precompute all scores and densities
    score = np.full((lengthS, lengthT + skip - 1), inf)
    for t in range(0, lengthT):
      for s in range(0, lengthS):
        score[s][t + skip - 1] = scores[start + t, hmm[s] / self.nstates]

    # forward
    scores = score[0, 0 + skip - 1:skip + skip - 2]
    scores = np.add(scores, tdps[1:])
    fwdScore[0, 0 + skip - 1:skip + skip - 2] = scores
    bt[0, 0 + skip - 1:skip + skip - 2] = range(1, skip)

    # remaining columns
    for s in range(1, lengthS):
      for t in range(max(lengthT - (lengthS - s) * skip,0), lengthT - (lengthS - s - 1)):
        previous = fwdScore[s - 1, t:t + skip]
        scores = score[s, t + skip - 1]
        scores = np.add(scores, np.add(previous, tdps[::-1]))
        fwdScore[s, t + skip - 1] = -logsumexp(-scores)

    score = score[:, skip - 1:lengthT + skip - 1]
    score = np.pad(score, pad_width=((0, 0), (0, skip - 1)), mode='constant',
                   constant_values=inf)

    # backward
    # initalize last columns, use lm of last character
    bwdScore[lengthS - 1][lengthT - 1] = score[lengthS - 1][lengthT - 1]
    # remaining columns
    for s in range(lengthS - 2, -1, -1):
      for t in range(0, lengthT):
        previous = bwdScore[s + 1, t:t + skip]
        scores = np.add(score[s, t], previous)
        bwdScore[s, t] = -logsumexp(-np.add(scores, np.asarray(self.tdps)))

    errScore = np.add(fwdScore, bwdScore)
    errScore = -np.logaddexp(-errScore, logsumexp(-errScore,axis=0))

    return errScore


class InvBacktrackOp(theano.Op):
  # Properties attribute
  __props__ = ('tdps', 'nstates', "penalty")

  # index_in, scores
  itypes = [theano.tensor.bmatrix, theano.tensor.ftensor3, theano.tensor.ftensor3]
  otypes = [theano.tensor.imatrix, theano.tensor.imatrix, theano.tensor.bmatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, scores, skips = inputs_storage[:3]
    transcript = np.zeros(index_in.shape,'int32')
    attention = np.zeros(index_in.shape, 'int32')
    index = np.zeros(index_in.shape, 'int8')
    for b in range(scores.shape[1]):
      length_x = index_in[:,b].sum()
      t, a = self._recognize(scores[:length_x, b], skips[:length_x, b])
      #if b == 0:
      #  print np.exp(-transitions[:length_x, b])
      length_y = len(a)
      transcript[:length_y, b] = t
      attention[:length_y, b] = np.asarray(a) + b * index_in.shape[0]
      index[:length_y, b] = 1

    output_storage[0][0] = transcript
    output_storage[1][0] = attention
    output_storage[2][0] = index

  def __init__(self, tdps, nstates, penalty):
    self.nstates = nstates
    self.penalty = penalty
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0], output_grads[1], output_grads[2]]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[0], input_shapes[0], input_shapes[0]]

  def _buildHmm(self, transcription):
    transcriptionLength = transcription.shape[0]
    hmm = np.zeros(transcriptionLength * self.nstates, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.nstates + 1
      for s in range(0, self.nstates):
        hmm[i * self.nstates + s] = startState + s - 1

    return hmm

  def _recognize2(self, scores, transitions):
    lengthT = scores.shape[0]
    transcript = []
    attention = []
    t = lengthT - 1
    while t >= 0:
      label = scores[t].argmin()
      if label % self.nstates == self.nstates - 1:
        if True or len(transcript) == 0 or label / self.nstates != transcript[-1]:
          attention.append(t)
          transcript.append(label / self.nstates)
      t -= transitions[t].argmin() + 1
    return transcript[::-1], attention[::-1]

  def _recognize(self, scores, skips):
    lengthT = scores.shape[0]
    transcript = []
    attention = []
    t = lengthT - 1
    n = 0
    m = lengthT
    lsc = 0
    #print skips.argmin(axis=1)
    while t >= 0:
      label = scores[t].argmin()
      lsc += scores[t]
      if n % self.nstates == 0:
        attention.append(t)
        #transcript.append(np.sum(scores[t:m]).argmin())
        #m = t
        transcript.append(lsc.argmin())
        lsc = 0
      t -= (skips[t,1:].argmin() + 1) #* 3 + 1
      n += 1
    return transcript[::-1], attention[::-1]

  def _recognize3(self, scores, transitions):
    lengthT = scores.shape[0]
    lengthS = transitions.shape[1]

    cost = np.full((lengthT, lengthT), np.inf, 'float32')
    back = np.full((lengthT, lengthT), np.inf, 'int32')

    cost[0] = np.min(scores[0])
    back[0] = -1

    transcript = []
    attention = []

    for s in xrange(1, lengthT):
      for t in xrange(min(s * lengthS, lengthT)):
        #if s % self.nstates == 0: # end state

        cost[s, t] = np.min(scores[s])
        q = transitions[t].copy()
        q[:min(t,lengthS)] += cost[s - 1, t - min(t,lengthS) : t]
        back[s, t] = q.argmin() + 1
        cost[s, t] += q.min()

    t = lengthT - 1
    s = 1
    while t >= 0 and s < lengthT:
      if s % self.nstates == 0:
        attention.append(t)
        transcript.append(scores[t].argmin()  / self.nstates)
      t -= back[-s, t]
      s += 1
    return transcript[::-1], attention[::-1]


class InvDecodeOp(theano.Op):
  # Properties attribute
  __props__ = ('tdps', 'nstates', "penalty")

  # index_in, scores
  itypes = [theano.tensor.bmatrix, theano.tensor.ftensor3]
  otypes = [theano.tensor.imatrix, theano.tensor.imatrix, theano.tensor.bmatrix]

  # Python implementation:
  def perform(self, node, inputs_storage, output_storage):
    index_in, scores = inputs_storage[:2]
    transcript = np.zeros(index_in.shape, 'int32')
    attention = np.zeros(index_in.shape, 'int32')
    index = np.zeros(index_in.shape, 'int8')
    max_length_y = 0
    for b in range(scores.shape[1]):
      length_x = index_in[:, b].sum()
      t, a = self._recognize(0, length_x, scores[:length_x, b])
      length_y = len(a)
      transcript[:length_y, b] = t
      attention[:length_y, b] = np.asarray(a) + b * index_in.shape[0]
      index[:length_y, b] = 1
      max_length_y = max(length_y, max_length_y)

    output_storage[0][0] = transcript
    output_storage[1][0] = attention
    output_storage[2][0] = index

  def __init__(self, tdps, nstates, penalty):
    self.nstates = nstates
    self.penalty = penalty
    self.tdps = tuple(tdps)

  def grad(self, inputs, output_grads):
    return [output_grads[0] * 0, output_grads[1] * 0, output_grads[2] * 0]

  def infer_shape(self, node, input_shapes):
    return [input_shapes[0], input_shapes[0], input_shapes[0]]

  def _buildHmm(self, transcription):
    transcriptionLength = transcription.shape[0]
    hmm = np.zeros(transcriptionLength * self.nstates, dtype=np.int32)

    for i in range(0, transcriptionLength):
      startState = transcription[i] * self.nstates + 1
      for s in range(0, self.nstates):
        hmm[i * self.nstates + s] = startState + s - 1

    return hmm

  def _recognize(self, start, end, scores):
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

    leftStart = np.full((hmmLength, seqLength + skip - 1), 0, dtype=np.uint64)
    rightStart = np.full((hmmLength, seqLength + skip - 1), 0, dtype=np.uint64)

    leftEpoch = np.full((hmmLength, seqLength + skip - 1), - 1, dtype=np.int64)
    rightEpoch = np.full((hmmLength, seqLength + skip - 1), - 1, dtype=np.int64)

    bestWordEnds = np.full((int(maxDuration * seqLength), seqLength), - 1, dtype=np.int64)
    bestWordEndsStart = np.full((int(maxDuration * seqLength), seqLength), - 1, dtype=np.int64)
    bestWordEndsEpoch = np.full((int(maxDuration * seqLength), seqLength), - 1, dtype=np.int64)

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
         self.nstates] + self.penalty)), axis=0)

      bestWordEnd = bestWordEnd * self.nstates

      # two times min might be improved
      bestWordEndScore = np.min(np.concatenate(
        ([leftScore[0]],
         leftScore[self.nstates:hmmLength:
         self.nstates] + self.penalty)), axis=0)

      bestWordEnds[a] = bestWordEnd[skip - 1::]
      for t in range(0, seqLength):
        bestWordEndsStart[a][t] = leftStart[bestWordEnd[t + skip - 1]][t + skip - 1]
        bestWordEndsEpoch[a][t] = leftEpoch[bestWordEnd[t + skip - 1]][t + skip - 1]

      for s in range(0, hmmLength):
        # for silence allow special transitions (only 1)
        if s == 0:
          for t in range(0, seqLength - 1):
            if leftScore[0][t + skip - 1] < bestWordEndScore[t + skip - 1] or bestWordEnd[t + skip - 1] == 0:
              # inner silence
              rightScore[0][t + skip] = leftScore[0][t + skip - 1] + score[0][t + skip]
              rightStart[0][t + skip] = leftStart[0][t + skip - 1]
              rightEpoch[0][t + skip] = leftEpoch[0][t + skip - 1]
            else:
              # to silence
              # only transition 1 is allowed
              rightScore[0][t + skip] = bestWordEndScore[t + skip - 1] + score[0][t + skip]
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

            rightScore[s][t + skip - 1] = scores[bestChoice]
            rightStart[s][t + skip - 1] = leftStart[s - 1][t + skip - 1 - bestChoice]
            rightEpoch[s][t + skip - 1] = leftEpoch[s - 1][t + skip - 1 - bestChoice]

      # finish epoch
      leftScore, rightScore = rightScore, leftScore
      leftStart, rightStart = rightStart, leftStart
      leftEpoch, rightEpoch = rightEpoch, leftEpoch

      rightScore = np.full((hmmLength, seqLength + skip - 1), inf)
      rightStart = np.full((hmmLength, seqLength + skip - 1), start, dtype=np.uint64)
      rightEpoch = np.full((hmmLength, seqLength + skip - 1), - 1, dtype=np.int64)

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
