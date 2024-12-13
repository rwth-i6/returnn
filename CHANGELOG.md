# Changelog

This is a list of notable new features,
or any changes which could potentially break or change the behavior of existing setups.

This is intentionally kept short. For a full change log, just see the Git log.


## 2024-12-13: Bump min Python version from 3.7 to 3.8 ([issue #1326](https://github.com/rwth-i6/returnn/issues/1326))

This also drops support for TF 1.x.

## 2024-06-07: `VariableDataset`

Custom subdataset per subepoch based on user-provided function.
Can be useful for advanced training pipeline
where you create some HDFs on-the-fly (e.g. for alignments)
and then want to load them in later epochs.

## 2024-05-28: [`DistributeFilesDataset`](https://github.com/rwth-i6/returnn/blob/master/returnn/datasets/distrib_files.py) ([PR #1521](https://github.com/rwth-i6/returnn/pull/1521), [issue #1519](https://github.com/rwth-i6/returnn/issues/1519))

`DistributeDataset` together with `FileCache`
allows to train on very large datasets which do not fit on the local disk.
`DistributeDataset` operates on a list of files,
and for each sub-epoch will only select a subset of the files,
and `FileCache` will cache the files locally.

It also was specifically designed with distributed training in mind.
The distributed random_seed_offset method can be used,
but sharding is also supported ([PR #1538](https://github.com/rwth-i6/returnn/pull/1538)).

## 2023-11-28: [`lovely_tensors`](https://xl0.github.io/lovely-tensors/) support

Set `use_lovely_tensors = True` in the config.
Should always be safe to use, thus can always be enabled.

## 2023-11-09: `LearningRateControl` saves more meta info in learning-rate-file

Like effective learning rate (after `dynamic_learning_rate`),
training step, GPU, RETURNN version, etc.
(Was extended a bit over time.)

## 2024-01-09: [Train proc manager](https://github.com/rwth-i6/returnn/blob/master/returnn/util/train_proc_manager.py)

Set `use_train_proc_manager = True` (currently PyTorch only).
Should always be safe to use, thus can always be enabled.
Auto restart RETURNN on crashes under certain conditions
(e.g. it must have trained at least one epoch successfully since the most recent restart).

## 2023-12-30: PyTorch handle OOM for forwarding, auto-split batch

Set `forward_auto_split_batch_on_oom = True` in the config (currently PyTorch only).
Should always be safe to use, thus can always be enabled.
This went through several iterations of approaches,
stumbling through a number of CPython and PyTorch bugs,
e.g. [CPython #113939](https://github.com/python/cpython/issues/113939),
[PyTorch #18853](https://github.com/pytorch/pytorch/issues/18853),
[PyTorch #27600](https://github.com/pytorch/pytorch/issues/27600).

## 2023-12-23: PyTorch distributed training with param averaging

In the `torch_distributed` config dict: Set `"reduce_type": "param"` and `"param_sync_step": ...`.

## 2023-10-24: [`watch_memory`](https://github.com/rwth-i6/returnn/blob/master/returnn/util/watch_memory.py): watches memory of all procs

Set `watch_memory = True` in the config.
Should not influence anything, thus can always be enabled.

## 2023-10-03: RETURNN frontend (RF) native helpers ([PR #1403](https://github.com/rwth-i6/returnn/pull/1403))

## 2023-06-09: PyTorch distributed training ([PR #1335](https://github.com/rwth-i6/returnn/pull/1335), [issue #1332](https://github.com/rwth-i6/returnn/issues/1332))

`torch_distributed` config setting.
Using the official PyTorch `DistributedDataParallel`, i.e. synchronized accumulated gradients.
Each worker uses a different `random_seed_offset` for the dataset.

## 2023-05-15: PyTorch automatic mixed precision (AMP) support ([PR #1322](https://github.com/rwth-i6/returnn/pull/1322))

## 2023-04-03: PyTorch `preload_from_files` support ([PR #1292](https://github.com/rwth-i6/returnn/pull/1292))

## 2023-03-26: [`MultiProcDataset`](https://github.com/rwth-i6/returnn/blob/master/returnn/datasets/multi_proc.py)

## 2023-02-24: Make [`Tensor` and `Dim`](https://returnn.readthedocs.io/en/latest/getting_started/data.html) backend independent ([PR #1261](https://github.com/rwth-i6/returnn/pull/1261), [issue #1165](https://github.com/rwth-i6/returnn/issues/1165))

* Rename `Data` to `Tensor`, `DimensionTag` to `Dim`.
* Before, in our `Tensor`, the `placeholder` (now `raw_tensor`) was either None (as a template)
  or a TensorFlow tensor (`tf.Tensor`).
  Now it can support any raw tensor type.
* Now `Tensor` and `Dim` are moved to `returnn.tensor`.

## 2023-02-20: RETURNN frontend (RF) ([issue #1120](https://github.com/rwth-i6/returnn/issues/1120), [issue #1264](https://github.com/rwth-i6/returnn/issues/1264))

Modern alternative to the network dictionary to define models.
Using Python code to define the network,
very similar to how it is done in PyTorch or Keras or Flax.

This evolved from [`returnn_common.nn`](https://github.com/rwth-i6/returnn_common/tree/main/nn) ([example](https://github.com/rwth-i6/returnn_common/wiki/RETURNN-example-config)),
which provided already a very similar API.
But now, we build it such that we support multiple backends.
Specifically, the current supported (or planned) backends:

* PyTorch (fully supported)
* RETURNN network dictionary (TensorFlow) (fully supported)
  (copied the `returnn_common.nn` code)
  (this might be deprecated in the future)
* TensorFlow (directly) (mostly supported)
* NumPy (partially supported)
* JAX (planned)

## 2023-02-03: Use `black`, drop Python 2 support ([PR #1255](https://github.com/rwth-i6/returnn/pull/1255), [issue #487](https://github.com/rwth-i6/returnn/issues/487), [issue #1158](https://github.com/rwth-i6/returnn/issues/1158))

## 2022-10-24: Remove Theano backend ([PR #1164](https://github.com/rwth-i6/returnn/pull/1164))

## 2022-09-12: PyTorch backend started ([issue #1120](https://github.com/rwth-i6/returnn/issues/1120))

This evolved over time.
It was planned from the beginning
to support pure PyTorch models defined by the user
but also RETURNN frontend (RF) models.

## 2022-04-24: TF eager execution initial support

## 2022-02-11: TF loss auto-flatten optimization ([PR #906](https://github.com/rwth-i6/returnn/pull/906))

## 2021-09-12: TF generalized attention, `CumConcatLayer` ([PR #589](https://github.com/rwth-i6/returnn/pull/589), [issue #391](https://github.com/rwth-i6/returnn/issues/391))

Generalizes `SelfAttentionLayer` to allow for more custom variants.
The difficulties were to support this when being inside a `RecLayer`
and then when the optimization would move it outside the loop.
For this, we introduced `CumConcatLayer`.
[Example config for decoder self-attention](https://github.com/rwth-i6/returnn/issues/391#issuecomment-917517032).
This can also be used just in the encoder, i.e. outside a `RecLayer` anyway
via `ReinterpretDataLayer` to create a new dim tag.
[Example config for encoder self-attention](https://github.com/rwth-i6/returnn/issues/391#issuecomment-919873563).

## 2021-08-25: Explicit `Data` dimension tags ([PR #579](https://github.com/rwth-i6/returnn/pull/579))

`Data` (later called `Tensor`) has `dim_tags` (later called `dims`)
to describe the full shape, i.e. the dims of each axis.
These are `DimensionTag` (later `Dim`) objects.
Before this change, we already had dim tags but only for dynamic dims.
Now they are consistently used for all dims.
This makes everything more consistent and more in line
with other named tensors / named dimensions frameworks.

## 2021-06-11: [Behavior versions](https://returnn.readthedocs.io/en/latest/configuration_reference/behavior_version.html) ([PR #534](https://github.com/rwth-i6/returnn/pull/534), [issue #508](https://github.com/rwth-i6/returnn/issues/508))

Setting `behavior_version` in config control the behavior of RETURNN
and allows to update bad/buggy/broken behavior without changing behavior for existing setups.

## 2021-06-04: Start of [`returnn_common.nn`](https://github.com/rwth-i6/returnn_common/tree/main/nn)

Allows to define the RETURNN network dictionary using a more modern Python API,
very similar to PyTorch or Keras.
[Example](https://github.com/rwth-i6/returnn_common/wiki/RETURNN-example-config).
Note that this later got merged into RETURNN frontend (RF).

## 2021-03-18: Subnetwork sub layer can be independent ([#473](https://github.com/rwth-i6/returnn/pull/473))

This has an effect on recurrent subnetworks.
In the optimization phase, individual sub layers can be optimized out of the loop now.
This is crucial to allow for an easy use of nested subnetworks.
Nested subnetworks are important to allow for generic building blocks
such as in the [returnn_common](https://github.com/rwth-i6/returnn_common) recipes.
This was a larger internal change in RETURNN,
which possibly can simplify other code in RETURNN like losses in subnetworks.

## 2021-03-08: Packed arrays ([#466](https://github.com/rwth-i6/returnn/issues/466), [#467](https://github.com/rwth-i6/returnn/pull/467), [#469](https://github.com/rwth-i6/returnn/pull/469)) and extended batch information (`BatchInfo`)

The extended batch information (`BatchInfo` attached to `Data`)
contains information about merged or packed dimensions in the batch dimension,
such as a beam (from beam search), fixed dimensions or variable-length dimensions.
This has an effect on keeping the information of beam search,
on `FlattenBatchLayer`, `SplitBatchTimeLayer`, `MergeDimsLayer` (on batch dim) and related.

## 2021-03-05: Fast literal Python parsing via native `literal_py_to_pickle`

We use literal Python format as serialization format in many places,
e.g. `OggZipDataset`.
The idea was that Python should be very fast in parsing Python code
(e.g. via `eval` or `ast.literal_eval`).
Unfortunately, it turned out that Python is not very fast at this
(specifically to parse *literal Python*, a subset of Python),
and e.g. JSON parsing is much faster.
We now have native code to parse literal Python,
which is much faster than before,
and this is already used in `OggZipDataset`.
Everything should work as before but just be faster.
Note that for the future,
it is probably a better idea to use JSON for serialization,
or some binary format.

## 2021-03-03: Simplified [`logging`](https://docs.python.org/3/library/logging.html) usage

## 2021-03-01: External module import with `import_` ([#436](https://github.com/rwth-i6/returnn/discussions/436))

Together with this mechanism, some common recipes are being developed
in [rwth-i6/returnn_common](https://github.com/rwth-i6/returnn_common).

## 2021-02-27: `SentencePieces` vocabulary class for [SentencePiece](https://github.com/google/sentencepiece/)

This can use BPE but also potentially better alternatives like unigram language model based subword units.
This can also do stochastic sampling for training.

## 2020-12-09: New `batch_norm` settings

We did not change the defaults.
However, we observed that the defaults don't make sense.
So if you have used `batch_norm` with the defaults before,
you likely want to redo any such experiments.
See [here](https://github.com/rwth-i6/pytorch-to-returnn/blob/a209cb6b2d43ae5a6dc46db42101b3c653dad03b/pytorch_to_returnn/torch/nn/modules/batchnorm.py#L97)
for reasonable defaults.
Esp you want to set `momentum` to a small number, like 0.1,
and you probably want `update_sample_only_in_training=True`
and `delay_sample_update=True`.

## 2020-11-06: [PyTorch-to-RETURNN project](https://github.com/rwth-i6/pytorch-to-returnn)

## 2020-08-14: GitHub actions CI to replace Travis ([PR #340](https://github.com/rwth-i6/returnn/pull/340), [issue #308](https://github.com/rwth-i6/returnn/issues/308))

## 2020-08-03: New code structure ([discussion](https://github.com/rwth-i6/returnn/issues/162))

`TFEngine` (or `returnn.TFEngine`) becomes `returnn.tf.engine`, etc.

## 2020-06-30: New generic training pipeline / extended custom pretraining ([discussion](https://github.com/rwth-i6/returnn/issues/311))

Define `def get_network(epoch: int, **kwargs): ...` in your config,
as an alternative to `pretrain` with custom `construction_algo` and `network`.
Otherwise this is pretty similar in behavior
(with all similar features, such as `#config` overwrites, dataset overwrites, etc),
but not treated as "pretraining",
but used always.

## 2020-06-12: TensorFlow 2 support ([discussion](https://github.com/rwth-i6/returnn/issues/283))

Configs basically should "just work".
We recommend everyone to use TF2 now.

## 2020-06-10: Distributed TensorFlow support ([discussion](https://github.com/rwth-i6/returnn/issues/296), [wiki](https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow))

See [`returnn.tf.distributed`](https://returnn.readthedocs.io/en/latest/api/tf.distributed.html).

## 2020-06-05: New TF dataset pipeline via `tf.dataset` ([discussion](https://github.com/rwth-i6/returnn/issues/292))

Define `def dataset_pipeline(context: InputContext) -> tf.data.Dataset`
in your config.
See [`returnn.tf.data_pipeline`](https://returnn.readthedocs.io/en/latest/api/tf.data_pipeline.html).

## 2019-08-20: Pretrain `#config` can overwrite datasets (`train`, `dev`, `eval`)

## 2019-08-13: `Data` `batch_shape_meta` extra debug repr output

This will show the same information as before, but much more compact,
and also in addition the dimension tags (`DimensionTag`),
which also got improved in many further cases.

## 2019-08-07: Overlay nets (`extra_nets`)

You can have e.g. multiple additional networks which redefine
existing layers (they would automatically share params),
which can use different flags (e.g. enable the search flag).

## 2019-07: Multiple stochastic (latent) variables

It was designed to support this from the very beginning,
but the implementation was never fully finished for this.
Now examples like hard attention work.

## 2019-05: Better support for RETURNN as a framework

`pip install returnn`, and then `import returnn`.

## 2019-03-29: Remove hard Theano dependency

## 2019-03-24 and ongoing: Automatic linter checks

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

## 2018-08: Multi-GPU support via [Horovod](https://github.com/horovod/horovod)

## 2017-05: Flexible `RecLayer`, encoder-decoder attention, beam search (Albert Zeyer)

## 2016-12: Start on [TensorFlow](https://www.tensorflow.org/) support (Albert Zeyer)

Initial working support already finished within that month.
TF 0.12.0.

## 2015-07: Fast CUDA LSTM kernel (Paul Voigtlaender)
## 2015-03: `SprintDataset`, interface to [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/) (Albert Zeyer)
## 2015-01: Albert Zeyer joined
## ~2013-2014 (?): Patrick Doetsch started the project (Theano)
