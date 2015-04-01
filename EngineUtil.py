
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
  shape = [0, 0]  # time,batch
  for batch in batches:
    shape = [max(shape[0], batch.data_shape[0]), shape[1] + batch.data_shape[1]]
  if shape[1] == 0:
    return False, len(batches)
  assert shape[0] * shape[1] > 0

  device.alloc_data(shape + [dataset.num_inputs * dataset.window], dataset.get_max_ctc_length(), pad=pad_batches)

  offset_slice = 0
  for batch in batches:
    dataset.load_seqs(batch.start_seq, batch.end_seq)

    for seq in batch.seqs:
      o = seq.batch_frame_offset
      q = seq.batch_slice + offset_slice
      l = seq.frame_length
      assert o + l <= shape[0]
      assert q < shape[1]
      device.index[o:o + l, q] = numpy.ones((l,), dtype='int8')

      with dataset.lock:
        data = dataset.get_data(seq.seq_idx)[seq.seq_start_frame:seq.seq_end_frame]
        targets = dataset.get_targets(seq.seq_idx)[seq.seq_start_frame:seq.seq_end_frame]
        device.data[o:o + l, q] = data
        device.targets[o:o + l, q] = targets

        if recurrent and pad_batches:
          assert o == 0  # Doesn't make sense otherwise.
          # pad with equivalent to 0
          # these are the hardcoded values for IAM
          # TODO load this from somewhere
          pad_data = [-1.46374, -0.151816, -0.161173, 0.0686325, 0.0231148, -0.154613,
                      -0.105614, 0.00550198, 0.0911985, 0.00502809, 0.0512826, -0.0181915,
                      0.0225053, -0.00149681, 0.0782062, 0.0412163, 0.0526166, -0.0722563,
                      0.0268245, -0.0277465, 0.258805, -0.187777, -2.3835, -1.42065]
          device.data[o + l:, q] = pad_data
          # also pad targets
          # hardcoded si for IAM
          # TODO load this from somewhere
          pad_target = 189
          device.targets[o + l:, q] = pad_target

        # Only copy ctc targets if chunking is inactive to avoid out of range access.
        # CTC is not compatible with chunking anyway.
        chunking_active = dataset.chunk_size > 0
        if dataset.has_ctc_targets() and not chunking_active:
          assert dataset.get_seq_length(seq.seq_idx) == l  # Full seq.
          device.ctc_targets[q] = dataset.get_ctc_targets(seq.seq_idx)

        device.tags[q] = dataset.get_tag(seq.seq_idx)

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
  batch.add_frames(seq_idx=seq, seq_start_frame=0, length=dataset.get_seq_length(seq))
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
