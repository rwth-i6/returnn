
import numpy
from EngineBatch import Batch
from Log import log


def assign_dev_data(device, dataset, batches, recurrent=False, pad_batches=False, exclude=None):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :type batches: list[EngineBatch.Batch]
  :type recurrent: bool
  :type pad_batches: bool
  :returns successful and how much batch idx to advance.
  :rtype: (bool,int)
  """
  if not exclude: exclude = []
  # The final device.data.shape is in format (time,batch,feature).
  shape = [0, 0]  # time,batch
  for batch in batches:
    shape = [max(shape[0], batch.data_shape[0]), shape[1] + batch.data_shape[1]]
  if shape[1] == 0:
    return False, len(batches)
  assert shape[0] * shape[1] > 0

  output_shape = { k : shape[:] for k in dataset.num_outputs }
  for k in output_shape:
    if dataset.get_target_dim(k) > 1:
      output_shape[k] += [dataset.get_target_dim(k)]

  device.alloc_data(shape + [dataset.num_inputs * dataset.window], output_shape, dataset.targets, dataset.get_max_ctc_length(), pad=pad_batches)

  offset_slice = 0
  for batch in batches:
    dataset.load_seqs(batch.start_seq, batch.end_seq)
    device.num_frames += batch.get_total_num_frames()
    for seq in batch.seqs:
      o = seq.batch_frame_offset
      q = seq.batch_slice + offset_slice
      l = seq.frame_length
      #assert o + l[0] <= shape[0]
      assert q < shape[1]
      device.input_index[o:o + l[0], q] = numpy.ones((l[0],), dtype='int8')
      device.output_index[o:o + l[1], q] = numpy.ones((l[1],), dtype='int8')

      with dataset.lock:
        data = dataset.get_data(seq.seq_idx)
        device.data[o:o + l[0], q] = data[seq.seq_start_frame[0]:seq.seq_end_frame[0]]
        for target in dataset.targets:
          targets = dataset.get_targets(target, seq.seq_idx)
          if targets is not None:
            device.targets[target][o:o + l[1], q] = targets[seq.seq_start_frame[1]:seq.seq_end_frame[1]]
            #if exclude:
            #  for i in xrange(l[1]):
            #    if device.targets[target][o + i, q] in exclude:
            #      device.index[o + i, q] = 0
        # Only copy ctc targets if chunking is inactive to avoid out of range access.
        # CTC is not compatible with chunking anyway.
        chunking_active = dataset.chunk_size > 0
        if dataset.has_ctc_targets() and not chunking_active:
          assert dataset.get_seq_length(seq.seq_idx) == l  # Full seq.
          device.ctc_targets[q] = dataset.get_ctc_targets(seq.seq_idx)

        device.tags[q] = dataset.get_tag(seq.seq_idx)

    #for i in xrange(device.input_index.shape[0]):
    #  if numpy.sum(device.input_index[i,:]) == 0:
    #    device.input_index[i,0] = 1
    #for i in xrange(device.output_index.shape[0]):
    #  if numpy.sum(device.output_index[i,:]) == 0:
    #    device.output_index[i,0] = 1
    # Note on multiple batches for the non-recurrent case:
    # We could either concatenate all into a single slice, or do multiple slices.
    # We do multiple slices here.
    # See also the `shape` calculation above.
    offset_slice += batch.num_slices

  return True, len(batches)


def assign_dev_data_single_seq(device, dataset, seq):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :param int seq: sorted seq idx
  :return: whether we succeeded
  :rtype: bool
  """
  batch = Batch()
  batch.add_frames(seq_idx=seq, seq_start_frame=numpy.array([0,0]), length=dataset.get_seq_length(seq))
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
