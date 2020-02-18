.. _configuration_pretraining:

===========
Pretraining
===========

The parameter to enable the pretraining is called **pretrain** and needs to be set to a dictionary.
The dictionary usually contains:

construction_algo
    this needs to be a function that has the signature (idx, net_dict) -> net_dict
    and should return the transformed network structure for pretraining step "idx".

repetitions
    this number defines how many epochs should be run during each pretraining step.
    If not specified this will be 1.
