
"""
Contains utility functions to construct a batch.
This is used both by Theano and TF.
"""

from __future__ import print_function

import numpy
from returnn.engine.batch import Batch
from returnn.log import log


def assign_dev_data(device, dataset, batches, load_seqs=True):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :type batches: list[EngineBatch.Batch]
  :param bool load_seqs:
  :returns successful and how much batch idx to advance.
  :rtype: (bool,int)
  """
  from returnn.datasets.basic import shapes_for_batches
  shapes = shapes_for_batches(batches, data_keys=device.used_data_keys, dataset=dataset)
  if shapes is None:
    return False, len(batches)
  device.alloc_data(shapes=shapes)
  offset_slice = 0

  for batch in batches:
    if load_seqs:
      dataset.load_seqs(batch.start_seq, batch.end_seq)
    device.num_frames += batch.get_total_num_frames()
    with dataset.lock:
      for seq in batch.seqs:
        o = seq.batch_frame_offset
        q = seq.batch_slice + offset_slice
        length = seq.frame_length
        # input-data, input-index will also be set in this loop. That is data-key "data".
        # targets are usually data-key "classes".
        for k in device.used_data_keys:
          # device.used_data_keys are set by the train-net, but we will also get here during forward-only,
          # e.g. via SprintInterface, where we don't have e.g. the "classes" data.
          # In that case, l.get(k) should be None. In some earlier code, l.get(k) could also be 0 in that case.
          if length.get(k) in [0, None]:
            continue
          data = dataset.get_data_slice(seq.seq_idx, k, seq.seq_start_frame[k], seq.seq_end_frame[k])
          ls = data.shape[0]
          if "[sparse:" in k:
            assert o[k] == 0, "sparse non-recurrent batching + chunking not implemented"
            _device_maybe_enlarge_data(device, k, ls)
          else:
            if ls != length[k]:
              raise Exception("got shape[0]: %i, expected: %i, start/end: %r/%r, seq_idx: %i, seq len: %r" % (
                ls, length[k], seq.seq_start_frame, seq.seq_end_frame, seq.seq_idx,
                dataset.get_seq_length(seq.seq_idx)))
          device.output_index[k][o[k]:o[k] + ls, q] = numpy.ones((ls,), dtype='int8')
          device.targets[k][o[k]:o[k] + ls, q] = data
        # Only copy ctc targets if chunking is inactive to avoid out of range access.
        # CTC is not compatible with chunking anyway.
        chunking_active = dataset.chunk_size != 0
        if dataset.has_ctc_targets() and not chunking_active:
          device.ctc_targets[q] = dataset.get_ctc_targets(seq.seq_idx)

        device.tags[q] = dataset.get_tag(seq.seq_idx)
    # Note on multiple batches for the non-recurrent case:
    # We could either concatenate all into a single slice, or do multiple slices.
    # We do multiple slices here.
    # See also the `shape` calculation above.
    offset_slice += batch.num_slices

  return True, len(batches)


def _device_maybe_enlarge_data(device, key, needed_len):
  cur_len = device.output_index[key].shape[0]
  if cur_len >= needed_len:
    return
  diff_len = needed_len - cur_len
  new_len = cur_len + int(diff_len * 1.5)  # a bit more than needed
  assert new_len >= needed_len
  # Also see Device.alloc_data() for reference.
  # First, new output_index.
  old_index = device.output_index[key]
  index_shape = list(old_index.shape)
  index_shape[0] = new_len
  device.output_index[key] = numpy.zeros(index_shape, dtype='int8')
  device.output_index[key][0:cur_len] = old_index
  # Now, new targets.
  old_targets = device.targets[key]
  targets_shape = list(old_targets.shape)
  targets_shape[0] = new_len
  device.targets[key] = numpy.full(targets_shape, -1, dtype=device.targets[key].dtype)
  device.targets[key][0:cur_len] = old_targets


def assign_dev_data_single_seq(device, dataset, seq, load_seqs=True):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :param int seq: sorted seq idx
  :param bool load_seqs:
  :return: whether we succeeded
  :rtype: bool
  """
  batch = Batch()
  batch.init_with_one_full_sequence(seq_idx=seq, dataset=dataset)
  success, _ = assign_dev_data(device, dataset, [batch], load_seqs=load_seqs)
  return success


def maybe_subtract_priors(network, train, config):
  """
  :type network: Network.LayerNetwork
  :type train: Dataset.Dataset
  :type config: returnn.config.Config
  """
  if config.bool('subtract_priors', False):
    prior_scale = config.float('prior_scale', 0.0)
    priors = train.calculate_priori()
    priors[priors == 0] = 1e-10  # avoid priors of zero which would yield a bias of inf
    ls = [p for p in network.train_params_vars if p.name == 'b_output']
    assert len(ls) == 1, len(ls)
    b_softmax = ls[0]
    b_softmax.set_value(b_softmax.get_value() - prior_scale * numpy.log(priors))
    print("subtracting priors with prior_scale", prior_scale, file=log.v3)
