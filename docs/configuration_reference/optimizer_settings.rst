.. _optimizer_settings:

==================
Optimizer Settings
==================

.. note::
    To define the update algorithm, there are two different methods. One is to set the desired algorithm explicitely,
    e.g. ``adam = True`` or ``rmsprop = True``. The other method is to set the parameter ``optimizer``
    and define the type by setting ``class`` in a dictionary. Currently available updater are:

        - adam
        - nadam
        - adadelta
        - adagrad
        - rmsprop

    if no updater is specified, SGD is used.

accum_grad_multiple_step
    An integer specifying the number of updates to stack the gradient, called "gradient accumulation".

adam
    Set to ``True`` to enable adam gradient updating.

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

optimizer_epsilon

user_learning_rate_control_always

meta_losses
    A list of (name, formula) tuples that define arbitrary meta losses.
    Every meta loss may combine any number of other losses. The formula is a string, which will be eval'ed.
    This can be used in combination with learning_rate_control_error_measure to make the LR scheduling dependant on multiple losses.



