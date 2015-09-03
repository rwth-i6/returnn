
import numpy
from EngineBatch import Batch
from Log import log
from Util import NumbersDict


def assign_dev_data(device, dataset, batches):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :type batches: list[EngineBatch.Batch]
  :returns successful and how much batch idx to advance.
  :rtype: (bool,int)
  """
  shapes = dataset.shapes_for_batches(batches, data_keys=device.used_data_keys)
  if shapes is None:
    return False, len(batches)
  import time
  ts = time.time()
  device.alloc_data(shapes=shapes, max_ctc_length=dataset.get_max_ctc_length())
  ts = time.time()
  offset_slice = 0

  for batch in batches:
    dataset.load_seqs(batch.start_seq, batch.end_seq)
    device.num_frames += batch.get_total_num_frames()
    with dataset.lock:
      for seq in batch.seqs:
        o = seq.batch_frame_offset
        q = seq.batch_slice + offset_slice
        l = seq.frame_length
        #assert o + l[0] <= shape[0]
        #assert q < shape[1]
        device.input_index[o["data"]:o["data"] + l["data"], q] = numpy.ones((l["data"],), dtype='int8')
        data = dataset.get_data(seq.seq_idx, "data")
        device.data[o["data"]:o["data"] + l["data"], q] = data[seq.seq_start_frame["data"]:seq.seq_end_frame["data"]]
        for k in device.used_data_keys:
          targets = dataset.get_data(seq.seq_idx, k)
          if targets is not None:
            device.output_index[k][o[k]:o[k] + l[k], q] = numpy.ones((l[k],), dtype='int8')
            device.targets[k][o[k]:o[k] + l[k], q] = targets[seq.seq_start_frame[k]:seq.seq_end_frame[k]]
            #if exclude:
            #  for i in xrange(l[1]):
            #    if device.targets[target][o + i, q] in exclude:
            #      device.index[o + i, q] = 0
        # Only copy ctc targets if chunking is inactive to avoid out of range access.
        # CTC is not compatible with chunking anyway.
        chunking_active = dataset.chunk_size > 0
        if dataset.has_ctc_targets() and not chunking_active:
          #assert dataset.get_seq_length_2d(seq.seq_idx) == l  # Full seq.
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
  batch.add_frames(seq_idx=seq, seq_start_frame=0, length=dataset.get_seq_length(seq))
  success, _ = assign_dev_data(device, dataset, [batch])
  return success


def maybe_subtract_priors(network, train, config):
  """
  :type network: Network.LayerNetwork
  :type train: Dataset.Dataset
  :type config: Config.Config
  """
  if config.bool('subtract_priors', False):
    prior_scale = config.float('prior_scale', 0.0)
    priors = train.calculate_priori()
    priors[priors == 0] = 1e-10 #avoid priors of zero which would yield a bias of inf
    l = [p for p in network.train_params_vars if p.name == 'b_output']
    assert len(l) == 1, len(l)
    b_softmax = l[0]
    b_softmax.set_value(b_softmax.get_value() - prior_scale * numpy.log(priors))
    print >> log.v3, "subtracting priors with prior_scale", prior_scale
