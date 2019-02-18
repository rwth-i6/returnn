# Changelog

This is a list of notable new features,
or any changes which could potentially break or change the behavior of existing setups.

This is intentionally kept short. For a full change log, just see the Git log.


## 2019-02-18: `MergeDimsLayer`, `SplitBatchTimeLayer`

If you used `MergeDimsLayer` with `"axes": "BT"` **on some time-major input**,
and then later `SplitBatchTimeLayer` to get the time-axis back, it was likely incorrect.

## 2019-02-09: `LayerBase` `updater_opts`, individual updater/optimizer options per layer

## 2019-01-30: video: RETURNN overview

## 2018-08: multi-GPU support via [Horovod](https://github.com/horovod/horovod)

## 2017-05: flexible `RecLayer`, encoder-decoder attention, beam search (Albert Zeyer)

## 2016-12: start on [TensorFlow](https://www.tensorflow.org/) support (Albert Zeyer)

Initial working support already finished within that month.
TF 0.12.0.

## 2015-07: fast CUDA LSTM kernel (Paul Voigtlaender)
## 2015-03: `SprintDataset`, interface to [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/) (Albert Zeyer)
## 2015-01: Albert Zeyer joined
## ~2013-2014 (?): Patrick Doetsch started the project (Theano)
