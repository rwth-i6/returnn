.. _dataset_reference:

========
Datasets
========

All datasets in RETURNN are based on the :class:`Datset.Dataset`,
and most are also based on either :class:`CachedDataset.CachedDataset` or :class:`CachedDataset2.CachedDataset2`.
The common parameters that can be used across most datasets are:

    - ``partition_epoch``: split the data into smaller parts per epoch
    - ``seq_ordering``: define the sequence ordering of the data.

Possible values for the sequence ordering are:

    - ``default``: Keep the sequences as is
    - ``reverse``: Use the default sequences in reversed order
    - ``random``: Shuffle the data with a predefined fixed seed
    - ``random:<seed>``: Shuffle the data with the seed given
    - ``sorted``: Sort by length (only if available), beginning with shortest sequences
    - ``sorted_reverse``: Sort by length, beginning with longest sequences
    - ``laplace:<n_buckets>``: Shuffle the data and sort by length within each of n bins, each second bin is sorted in reverse.
    - ``laplace:.<n_sequences>``: As above, but the number of bins is chosen such that each bin contains roughly n sequences.
    - ``laplace:<n_buckets>:<seed>``: A seed can be provided for both laplace versions, separated by an additional colon.

Note that not all sequence order modes are available for all datasets,
and some datasets may provide additional modes.
For details on the different sequence orderings, have a look at :func:`Dataset.Dataset.get_seq_order_for_epoch`.
Also check the sequence ordering possibilities with the :ref:`MetaDataset <meta_dataset>`.



.. toctree::
    :maxdepth: 1
    :titlesonly:

    generic_datasets.rst
    text_datasets.rst
    audio_datasets.rst
    combination.rst


.. autoclass:: Dataset.Dataset
    :show-inheritance:

.. autoclass:: CachedDataset.CachedDataset
    :show-inheritance:

.. autoclass:: CachedDataset2.CachedDataset2
    :show-inheritance:
