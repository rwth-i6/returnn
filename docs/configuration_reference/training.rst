.. _configuration_training:

========
Training
========

batch_size
    An integer defining the batch size in data items (frames, words, subwords, etc.) per batch.
    A mini-batch has at least a time-dimension and a batch-dimension (or sequence-dimension),
    and depending on dense or sparse, also a feature-dimension.
    ``batch_size`` is the upper limit for ``time * sequences`` during creation of the mini-batches.

batching

chunking
    You can chunk sequences of your data into parts, which will greatly reduce the amount of needed zero-padding.
    This option is a string of two numbers, separated by a comma, i.e. ``chunk_size:chunk_step``,
    where ``chunk_size`` is the size of a chunk,
    and ``chunk_step`` is the step after which we create the next chunk.
    I.e. the chunks will overlap by ``chunk_size - chunk_step`` frames.
    Set this to ``0`` to disable it, or for example ``100:75`` to enable it.

cleanup_old_models
    If set to ``True``, checkpoints are removed based on their score on the dev set.
    Per default, 2 recent, 4 best, and the checkpoints 20,40,80,160,240 are kept.
    Can be set as a dictionary to specify additional options.

        - ``keep_last_n``: integer defining how many recent checkpoints to keep
        - ``keep_best_n``: integer defining how many best checkpoints to keep
        - ``keep``: list or set of integers defining which checkpoints to keep

max_seq_length
    A dict with string:integer pairs. The string must be a valid data key,
    and the integer specifies the upper bound for this data object. Batches, where the specified data object exceeds
    the upper bound are discarded. Note that some datasets (e.g ``OggZipDataset``) load and process the data
    to determine the length, so even for discarded sequences data processing might be performed.

max_seqs
    An integer specifying the upper limit of sequences in a batch (can be used in addition to ``batch_size``).

num_epochs
    An integer specifying the number of epochs to train.

save_interval
    An integer specifying after how many epochs the model is saved.

start_epoch
    An integer or string specifying the epoch to start the training at. The default is 'auto'.

stop_on_nonfinite_train_score
    If set to ``False``, the training will not be interupted if a single update step has a loss with NaN of Inf









