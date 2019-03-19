.. _dataset_combination:

===================
Dataset Combination
===================


Meta Dataset
------------

The MetaDataset (API: :ref:`meta_dataset`) is to be used in the case of **Multimodality**.
Here, the datasets are expected to describe different features of the **same training sequences**.
These features will all be available to the network at the same time.

The datasets to be combined are given via the input parameter ``"datasets"``.
To define which training examples from the different datasets belong together, a ``"seq_list_file"`` in pickle format has to be created.
It contains a list of sequence tags for each dataset (see example below).
Note, that in general each dataset type has its own tag format, e.g. for the TranslationDataset it is ``line-<n>``, for the SprintDataset it is ``<corpusname>/<recording>/<segment id>``.
**Providing a sequence list can be omitted**, if the set of sequence tags is the same for all datasets.
When using multiple ExternSprintDataset instances, the sprint segment file can be provided as sequence list.
In this case the MetaDataset assumes that the sequences with equal tag correspond to each other.
This e.g. works when combining TranslationDatasets if all the text files are sentence aligned.


**Example of Sequence List:**

.. code::

    { 'sprint': [
        'corpus/ted_1/1',
        'corpus/ted_1/2',
        'corpus/ted_1/3',
        'corpus/ted_1/4',
    'translation': [
        'line-0',
        'line-1',
        'line-2',
        'line-3']
    }

Python dict stored in pickle file. E.g. the sequence tagged with 'corpus/ted_1/3' in the 'sprint' dataset corresponds to the sequence tagged 'line-2'
in the 'translation' dataset.

**Example of MetaDataset config:**

.. code::

    train = {"class": "MetaDataset", "seq_list_file": "seq_list.pkl",
             "datasets": {"sprint": train_sprint, "translation": train_translation},
             "data_map": {"data": ("sprint", "data"),
             "target_text_sprint": ("sprint", "orth_classes"),
             "source_text": ("translation", "data"),
             "target_text": ("translation", "classes")},
             "seq_ordering": "random",
             "partition_epoch": 2,
    }

This combines a SprintDataset and a TranslationDataset. These are defined as ``"train_sprint"`` and ``"train_translation"`` separately.
*Note that the current implementation expects one input feature to be called "data".*

Combined Dataset
----------------

The CombinedDataset (API: :ref:`meta_dataset`) is to be used in the cases of **Multi-Task Learning** and **Combination of Corpora**.
Here, in general, the datasets describe **different training sequences**.
For each sequence, only the features of the corresponding dataset will be available.
Features of the other datasets are set to empty arrays.
The input parameter ``"datasets"`` is the same as for the MetaDataset.
The ``"data_map"`` is reversed to allow for several datasets mapping to the same feature.
The "default" ``"seq_ordering"`` is to first go through all sequences of the first dataset, then the second and so on.
All other sequence orderings (``"random"``, ``"sorted"``, ``"laplace"``, ...) are supported and based on this "default" ordering.
There is a special sequence ordering ``"random_dataset"``, where we pick datasets at random, while keeping the sequence order within the datasets as is.
To adjust the ratio of number of training examples from the different datasets in an epoch one can use "repeat_epoch" in some of the datasets to
increase their size relative to the others.
Also, ``"partition_epoch"`` in some of the datasets can be used to shrink them relative to the others.

**Example of CombinedDataset config:**

.. code::

    train = {"class": "CombinedDataset",
             "datasets": {"sprint": train_sprint, "translation": train_translation},
             "data_map": {("sprint", "data"): "data",
                          ("sprint", "orth_classes"): "orth_classes",
                          ("translation", "data"): "source_text",
                          ("translation", "classes"): "orth_classes"},
             "seq_ordering": "default",
             "partition_epoch": 2,
     }

This combines a SprintDataset and a TranslationDataset. These are defined as ``"train_sprint"`` and ``"train_translation"`` separately.
*Note that the current implementation expects one input feature to be called "data".*
