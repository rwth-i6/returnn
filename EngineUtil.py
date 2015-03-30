
import numpy
from EngineBatch import Batch
from Log import log


def assign_dev_data(device, dataset, batches, recurrent=False, pad_batches=False):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :type batches: list[EngineBatch.Batch]
  :type recurrent: bool
  :type pad_batches: bool
  :returns successful and how much batch idx to advance.
  :rtype: (bool,int)
  """
  # The final device.data.shape is in format (time,batch,feature).
  shape = [0, 0]
  for batch in batches:
    shape = [max(shape[0], batch.data_shape[0]), shape[1] + batch.data_shape[1]]
  if shape[1] == 0:
    return False, len(batches)

  device.alloc_data(shape + [dataset.num_inputs * dataset.window], dataset.max_ctc_length, pad=pad_batches)
  offset = 0
  for i, batch in enumerate(batches):
    if not dataset.have_seqs(batch.start[0], batch.get_end_seq()):
      # We could also just skip those seqs. However, we might want to keep all batches
      # of similar sizes to have more stable training. Thus, we skip this batch.
      return False, i + 1

    dataset.load_seqs(batch.start[0], batch.get_end_seq())
    idi = dataset.alloc_interval_index(batch.start[0])
    assert idi >= 0, "failed to load seqs (%i, %i)" % (batch.start[0], batch.get_end_seq())
    if recurrent:
      for s in xrange(batch.start[0], batch.start[0] + batch.data_shape[1]):
        ids = dataset.seq_index[s]  # the real seq idx after sorting
        l = dataset.seq_lengths[ids]
        with dataset.lock:
          o = dataset.seq_start[s] + batch.start[1] - dataset.seq_start[dataset.alloc_intervals[idi][0]]
          assert o >= 0
          q = s - batch.start[0] + offset
          device.data[:l, q] = dataset.alloc_intervals[idi][2][o:o + l]
        device.targets[:l, q] = dataset.targets[dataset.seq_start[s] + batch.start[1]:dataset.seq_start[s] + batch.start[1] + l]
        if pad_batches:
          #pad with equivalent to 0
          #these are the hardcoded values for IAM
          #TODO load this from somewhere
          pad_data = [-1.46374, -0.151816, -0.161173, 0.0686325, 0.0231148, -0.154613,
                      -0.105614, 0.00550198, 0.0911985, 0.00502809, 0.0512826, -0.0181915,
                      0.0225053, -0.00149681, 0.0782062, 0.0412163, 0.0526166, -0.0722563,
                      0.0268245, -0.0277465, 0.258805, -0.187777, -2.3835, -1.42065]
          device.data[l:, q] = pad_data
          #also pad targets
          #hardcoded si for IAM
          #TODO load this from somewhere
          pad_target = 189
          device.targets[l:, q] = pad_target
        #only copy ctc targets if chunking is inactive to avoid out of range access (ctc is not comaptible with chunking anyway)
        chunking_active = dataset.chunk_size > 0
        if dataset.ctc_targets is not None and not chunking_active:
          device.ctc_targets[q] = dataset.ctc_targets[ids]
        device.tags[q] = dataset.tags[ids] #TODO
        device.index[:l, q] = numpy.ones((l,), dtype = 'int8')
      offset += batch.data_shape[1]
    else:
      with dataset.lock:
        seq_start = dataset.seq_start[batch.start[0]] + batch.start[1]
        alloc_start_seq, _, alloc_data = dataset.alloc_intervals[idi]
        o = seq_start - dataset.seq_start[alloc_start_seq]
        assert o >= 0
        l = batch.data_shape[0]
        assert alloc_data.shape[0] >= o + l
        device.data[offset:offset + l, 0] = alloc_data[o:o + l]
      device.targets[offset:offset + l, 0] = dataset.targets[seq_start:seq_start + l]
      device.index[offset:offset + l, 0] = numpy.ones((l,), dtype='int8')
      offset += l

  return True, len(batches)


def assign_dev_data_single_seq(device, dataset, seq):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :param int seq: sorted seq idx
  :return: whether we succeeded
  :rtype: bool
  """
  if not dataset.have_seqs(seq, seq + 1):
    return False
  batch = Batch([seq, 0])
  batch.data_shape = (dataset.get_seq_length(seq), 1)
  success, _ = assign_dev_data(device, dataset, [batch])
  return success


def subtract_priors(network, train, config):
  if config.bool('subtract_priors', False):
    prior_scale = config.float('prior_scale', 0.0)
    priors = train.calculate_priori()
    priors[priors == 0] = 1e-10 #avoid priors of zero which would yield a bias of inf
    l = [p for p in network.train_params_vars if p.name == 'b_output']
    assert len(l) == 1, len(l)
    b_softmax = l[0]
    b_softmax.set_value(b_softmax.get_value() - prior_scale * numpy.log(priors))
    print >> log.v3, "subtracting priors with prior_scale", prior_scale
