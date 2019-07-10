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

max_seq_length

max_seqs
    An integer specifying the upper limit of sequences in a batch (can be used in addition to ``batch_size``).

num_epochs
    An integer specifying the number of epochs to train.

save_interval
    An integer specifying after how many epochs the model is saved.










