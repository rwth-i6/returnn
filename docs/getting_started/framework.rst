.. framework:

====================
RETURNN as Framework
====================

Install RETURNN via ``pip`` (`PyPI entry <https://pypi.org/project/returnn/>`__).
Then :code:`import returnn` should work.
See `demo-returnn-as-framework.py <https://github.com/rwth-i6/returnn/blob/master/demos/demo-returnn-as-framework.py>`__ as a full example.

Basically you can write very high level code like this::

    from returnn.TFEngine import Engine
    from returnn.Dataset import init_dataset
    from returnn.Config import get_global_config

    config = get_global_config(auto_create=True)
    config.update(dict(
        # ...
    ))

    engine = Engine(config)

    train_data = init_dataset({"class": "Task12AXDataset", "num_seqs": 1000, "name": "train"})
    dev_data = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})

    engine.init_train_from_config(train_data=train_data, dev_data=dev_data)

Or you go lower level and construct the computation graph yourself::

    from returnn.TFNetwork import TFNetwork

    config = get_global_config(auto_create=True)

    net = TFNetwork(train_flag=True)
    net.construct_from_dict({
        # ...
    })
    fetches = net.get_fetches_dict()

    with tf.Session() as session:
        results = session.run(fetches, feed_dict={
            # ...
            # you could use FeedDictDataProvider
        })

Or even lower level and just use parts from ``TFUtil``, ``TFNativeOp``, etc.::

    from returnn.TFNativeOp import ctc_loss
    from returnn.TFNativeOp import edit_distance
    from returnn.TFNativeOp import NativeLstm2

    from returnn.TFUtil import ctc_greedy_decode
    from returnn.TFUtil import get_available_gpu_min_compute_capability
    from returnn.TFUtil import safe_log
    from returnn.TFUtil import reuse_name_scope
    from returnn.TFUtil import dimshuffle

    # ...
