# Changelog

This is a list of notable new features,
or any changes which could potentially break or change the behavior of existing setups.

This is intentionally kept short. For a full change log, just see the Git log.


## 2020-08-03: New code structure ([discussion](https://github.com/rwth-i6/returnn/issues/162))

## 2020-06-30: New generic training pipeline / extended custom pretraining ([discussion](https://github.com/rwth-i6/returnn/issues/311))

## 2020-06-12: TensorFlow 2 support ([discussion](https://github.com/rwth-i6/returnn/issues/283))

## 2020-06-10: Distributed TensorFlow support ([discussion](https://github.com/rwth-i6/returnn/issues/296), [wiki](https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow))

## 2020-06-05: New TF dataset pipeline via `tf.dataset` ([discussion](https://github.com/rwth-i6/returnn/issues/292))

## 2019-08-20: Pretrain `#config` can overwrite datasets (`train`, `dev`, `eval`)

## 2019-08-13: `Data` `batch_shape_meta` extra debug repr output

This will show the same information as before, but much more compact,
and also in addition the dimension tags (`DimensionTag`),
which also got improved in many further cases.

## 2019-08-07: overlay nets (`extra_nets`)

You can have e.g. multiple additional networks which redefine
existing layers (they would automatically share params),
which can use different flags (e.g. enable the search flag).

## 2019-07: multiple stochastic (latent) variables

It was designed to support this from the very beginning,
but the implementation was never fully finished for this.
Now examples like hard attention work.

## 2019-05: better support for RETURNN as a framework

`pip install returnn`, and then `import returnn`.

## 2019-03-29: remove hard Theano dependency

## 2019-03-24 and ongoing: automatic linter checks

Currently pylint and PyCharm inspection checks automatically run in Travis.
Both have some false positives, but so far the PyCharm inspections seems much more sane.
A lot of code cleanup is being done now.
This is not complete yet, and thus the failing tests are ignored.

## 2019-03-01: `GenericAttentionLayer` reimplemented

Based on `DotLayer` now.
Is more generic if the attention weights
have multiple time axes (e.g. in Transformer training).
Does checks whether the base time axis
and weights time axis match,
and should automatically select the right one from weights
if there are multiple
(before: it always used the first weights time axis).
The output format (order of axes) might be
different than it was before in some cases.

## 2019-03-01: `Data` some slight behavior changes

E.g. the default feature dim axis (if unspecified)
is the last non-dynamic axis.
Also in some cases the time axis will be
automatically re-selected if the original one
was removed and there are multiple dynamic axes.
`DimensionTag` support was extended.
When copying compatible to some other data
with multiple dynamic axes, it will more correctly
match the dynamic axes via the dimension tags
(see test cases for examples).

## 2019-03-01: `SqueezeLayer`, `enforce_batch_dim_axis` by default `None`

I.e. the output format (order of axes) might be
different than it was before in some cases.

## 2019-02-27: `CombineLayer` / `EvalLayer` / any which concatenate multiple sources, extended automatic broadcasting

See e.g. `concat_sources`.

## 2019-02-21: `HDFDataset` huge speedup for `cache_size=0`

If your whole dataset does not fit into memory
(or you don't want to consume so much memory),
for TensorFlow,
you should always use `cache_size = 0` (or `"0"`) in the config.
This case got a huge speedup.

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
