.. _optimizer_settings:

==================
Optimizer Settings
==================

.. note::
    To define the update algorithm, set the parameter ``optimizer`` to a dictionary
    and define the type by setting ``class``.
    All available optimizers and their parameters can be found :ref:`here <optimizer>`.
    Setting the learning rate should not set in the dict, but rather separately.
    If no updater is specified, plain SGD is used.

accum_grad_multiple_step
    An integer specifying the number of updates to stack the gradient, called "gradient accumulation".

gradient_clip
    Specifiy a gradient clipping threshold.

gradient_noise
    Apply a (gaussian?) noise to the gradient with given deviation (variance? stddev?)

learning_rate
    Specifies the global learning rate

learning_rate_control

learning_rate_control_error_measure

    A str to define which score or error is used to control the learning rate reduction. Per default, Returnn will use dev_score_output. A typical choice would be dev_score_LAYERNAME or dev_error_LAYERNAME. Can be set to None to disable learning rate control.

learning_rate_control_min_num_epochs_per_new_lr

learning_rate_control_relative_error_relative_lr

learning_rate_file
    A path to a file storing the learning rate for each epoch. Despite the name, also stores scores and errors.

learning_rates
    A list of learning rates that defines the learning rate for each epoch from the beginning.
    Can be used for learning-rate warmup.

newbob_learning_rate_decay

newbob_multi_num_epochs

newbob_multi_update_interval

newbob_relative_error_threshold

user_learning_rate_control_always





