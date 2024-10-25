#!/usr/bin/env python3

"""
Main entry point
================

This is the main entry point, providing :func:`main`.
See :func:`init_config` for some arguments, or just run ``./rnn.py --help``.
See :ref:`tech_overview` for a technical overview.
"""

from __future__ import annotations

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2014"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHOCR"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"


import os
import sys
import time
from typing import TYPE_CHECKING, Optional, Union, Any, Sequence, Dict
import numpy
import returnn
from returnn.log import log
from returnn.config import Config, get_global_config
from returnn.datasets import Dataset, init_dataset, init_dataset_via_str
from returnn.datasets.hdf import HDFDataset
from returnn.util import basic as util, debug as debug_util
from returnn.util.basic import BackendEngine, BehaviorVersion

# These imports are not directly used here, but make them available, as other code imports them from here.
# noinspection PyUnresolvedReferences
from returnn.util.debug import init_ipython_kernel, init_better_exchook, init_faulthandler, debug_shell

# noinspection PyUnresolvedReferences
from returnn.util.basic import init_thread_join_hack, describe_returnn_version

if TYPE_CHECKING:
    import returnn.tf.engine
    import returnn.torch.engine

config = None  # type: Optional[Config]
engine = None  # type: Optional[Union[returnn.tf.engine.Engine, returnn.torch.engine.Engine]]
train_data = None  # type: Optional[Dataset]
dev_data = None  # type: Optional[Dataset]
eval_data = None  # type: Optional[Dataset]
quit_returnn = False


def init_config(
    config_filename: Optional[str] = None,
    command_line_options: Sequence[str] = (),
    default_config: Optional[Dict[str, Any]] = None,
    extra_updates: Optional[Dict[str, Any]] = None,
):
    """
    :param config_filename:
    :param command_line_options: e.g. ``sys.argv[1:]``
    :param default_config:
    :param extra_updates:

    Initializes the global config.
    There are multiple sources which are used to init the config:

      * ``configFilename``, and maybe first item of ``commandLineOptions`` interpret as config filename
      * other options via ``commandLineOptions``
      * ``extra_updates``

    Note about the order/priority of these:

      * ``extra_updates``
      * options from ``commandLineOptions``
      * ``configFilename``
      * config filename from ``commandLineOptions[0]``
      * ``extra_updates``
      * options from ``commandLineOptions``

    ``extra_updates`` and ``commandLineOptions`` are used twice so that they are available
    when the config is loaded, which thus has access to them, and can e.g. use them via Python code.
    However, the purpose is that they overwrite any option from the config;
    that is why we apply them again in the end.

    ``commandLineOptions`` is applied after ``extra_updates`` so that the user has still the possibility
    to overwrite anything set by ``extra_updates``.
    """
    global config
    config = Config()

    config_filenames_by_cmd_line = []
    if command_line_options:
        # Assume that the first argument prefixed with "+" or "-" and all following is not a config file.
        i = 0
        for arg in command_line_options:
            if arg[:1] in "-+":
                break
            config_filenames_by_cmd_line.append(arg)
            i += 1
        command_line_options = command_line_options[i:]

    if default_config:
        config.update(default_config)
    if extra_updates:
        config.update(extra_updates)
    if command_line_options:
        config.parse_cmd_args(command_line_options)
    if config_filename:
        config.load_file(config_filename)
    for fn in config_filenames_by_cmd_line:
        config.load_file(fn)
    if extra_updates:
        config.update(extra_updates)
    if command_line_options:
        config.parse_cmd_args(command_line_options)

    # I really don't know where to put this otherwise:
    if config.bool("EnableAutoNumpySharedMemPickling", False):
        import returnn.util.task_system

        returnn.util.task_system.SharedMemNumpyConfig["enabled"] = True
    # Server default options
    if config.value("task", "train") == "server":
        config.set("num_inputs", 2)
        config.set("num_outputs", 1)

    BehaviorVersion.set(config.int("behavior_version", None))


def init_log():
    """
    Initializes the global :class:`Log`.
    """
    log.init_by_config(config)


def get_cache_byte_sizes():
    """
    :rtype: (int,int,int)
    :returns cache size in bytes for (train,dev,eval)
    """
    cache_sizes_user = config.list("cache_size", ["%iG" % util.default_cache_size_in_gbytes()])
    num_datasets = 1 + config.has("dev") + config.has("eval")
    cache_factor = 1.0
    if len(cache_sizes_user) == 1:
        cache_sizes_user *= 3
        cache_factor /= float(num_datasets)
    elif len(cache_sizes_user) == 2:
        cache_sizes_user.append("0")
    assert len(cache_sizes_user) == 3, "invalid amount of cache sizes specified"
    cache_sizes = []
    for cache_size_user in cache_sizes_user:
        cache_size = cache_factor * float(cache_size_user.replace("G", "").replace("M", "").replace("K", ""))
        assert len(cache_size_user) - len(str(cache_size)) <= 1, "invalid cache size specified"
        if cache_size_user.find("G") > 0:
            cache_size *= 1024 * 1024 * 1024
        elif cache_size_user.find("M") > 0:
            cache_size *= 1024 * 1024
        elif cache_size_user.find("K") > 0:
            cache_size *= 1024
        cache_size = int(cache_size) + 1 if int(cache_size) > 0 else 0
        cache_sizes.append(cache_size)
    return cache_sizes


# noinspection PyShadowingNames
def load_data(config, cache_byte_size, files_config_key, **kwargs):
    """
    :param Config config:
    :param int cache_byte_size:
    :param str files_config_key: such as "train" or "dev"
    :param kwargs: passed on to init_dataset() or init_dataset_via_str()
    :rtype: (Dataset,int)
    :returns the dataset, and the cache byte size left over if we cache the whole dataset.
    """
    if not config.bool_or_other(files_config_key, None):
        return None, 0
    kwargs = kwargs.copy()
    kwargs.setdefault("name", files_config_key)
    if config.is_typed(files_config_key) and isinstance(config.typed_value(files_config_key), dict):
        config_opts = config.typed_value(files_config_key)
        assert isinstance(config_opts, dict)
        kwargs.update(config_opts)
        if "cache_byte_size" not in config_opts:
            if kwargs.get("class", None) == "HDFDataset":
                kwargs["cache_byte_size"] = cache_byte_size
        Dataset.kwargs_update_from_config(config, kwargs)
        data = init_dataset(kwargs)
    elif config.is_typed(files_config_key) and callable(config.typed_value(files_config_key)):
        data = init_dataset(config.typed_value(files_config_key), default_kwargs=kwargs)
    else:
        config_str = config.value(files_config_key, "")
        data = init_dataset_via_str(config_str, config=config, cache_byte_size=cache_byte_size, **kwargs)
    cache_leftover = 0
    if isinstance(data, HDFDataset):
        cache_leftover = data.definite_cache_leftover
    return data, cache_leftover


def init_data():
    """
    Initializes the globals train,dev,eval of type Dataset.
    """
    cache_byte_sizes = get_cache_byte_sizes()
    global train_data, dev_data, eval_data
    dev_data, extra_cache_bytes_dev = load_data(
        config, cache_byte_sizes[1], "dev", **Dataset.get_default_kwargs_eval(config=config)
    )
    eval_data, extra_cache_bytes_eval = load_data(
        config, cache_byte_sizes[2], "eval", **Dataset.get_default_kwargs_eval(config=config)
    )
    train_cache_bytes = cache_byte_sizes[0]
    if train_cache_bytes >= 0:
        # Maybe we have left over cache from dev/eval if dev/eval have cached everything.
        train_cache_bytes += extra_cache_bytes_dev + extra_cache_bytes_eval
    train_data, extra_train = load_data(config, train_cache_bytes, "train")


def setup_dummy_datasets():
    """setup config to use :class:`DummyGenericDataset` instead of the normal datasets"""
    from binascii import crc32

    extern_data = config.typed_value("extern_data")
    assert extern_data, "must define extern_data to setup dummy datasets"
    num_seqs = config.int("dummy_dataset_num_seqs", None)
    if config.bool_or_other("train"):
        train_num_seqs = config.int("train_dummy_dataset_num_seqs", num_seqs)
        if not train_num_seqs:
            train_num_seqs = 1000
        if not num_seqs:
            num_seqs = max(train_num_seqs // 20, 1)
        config.set("train", {"class": "DummyGenericDataset", "data_template": extern_data, "num_seqs": train_num_seqs})
    if not num_seqs:
        num_seqs = 100
    for key in ["dev", "eval", "forward_data", "search_data"]:
        if config.bool_or_other(key):
            config.set(
                key,
                {
                    "class": "DummyGenericDataset",
                    "data_template": extern_data,
                    "num_seqs": num_seqs,
                    "fixed_random_seed": crc32(key.encode("utf8")),
                },
            )
    if config.bool_or_other("eval_datasets"):
        eval_datasets = config.typed_value("eval_datasets")
        assert isinstance(eval_datasets, dict)
        config.set(
            "eval_datasets",
            {
                key: {
                    "class": "DummyGenericDataset",
                    "data_template": extern_data,
                    "num_seqs": num_seqs,
                    "fixed_random_seed": crc32(key.encode("utf8")),
                }
                for key in eval_datasets.keys()
            },
        )


def print_task_properties():
    """
    print information about used data
    """
    if train_data:
        print("Train data:", file=log.v2)
        print("  input:", train_data.num_inputs, "x", train_data.window, file=log.v2)
        print("  output:", train_data.num_outputs, file=log.v2)
        print(" ", train_data.len_info(fast=True) or "no info", file=log.v2)
    if dev_data:
        print("Dev data:", file=log.v2)
        print(" ", dev_data.len_info(fast=True) or "no info", file=log.v2)
    if eval_data:
        print("Eval data:", file=log.v2)
        print(" ", eval_data.len_info(fast=True) or "no info", file=log.v2)


def init_engine():
    """
    Initializes global ``engine``, for example :class:`returnn.tf.engine.Engine`.
    """
    global engine
    if BackendEngine.is_tensorflow_selected():
        from returnn.tf.engine import Engine

        engine = Engine(config=config)
    elif BackendEngine.is_torch_selected():
        from returnn.torch.engine import Engine

        engine = Engine(config=config)
    else:
        raise NotImplementedError("Backend engine not implemented")


def returnn_greeting(config_filename=None, command_line_options=None):
    """
    Prints some RETURNN greeting to the log.

    :param str|None config_filename:
    :param list[str]|None command_line_options:
    """
    print(
        "RETURNN starting up, version %s, date/time %s, pid %i, cwd %s, Python %s"
        % (
            util.describe_returnn_version(),
            time.strftime("%Y-%m-%d-%H-%M-%S (UTC%z)"),
            os.getpid(),
            os.getcwd(),
            sys.executable,
        ),
        file=log.v3,
    )
    if config_filename:
        print("RETURNN config: %s" % config_filename, file=log.v4)
        if os.path.islink(config_filename):
            print("RETURNN config is symlink to: %s" % os.readlink(config_filename), file=log.v4)
    if command_line_options is not None:
        print("RETURNN command line options: %s" % (command_line_options,), file=log.v4)
    import socket

    print("Hostname:", socket.gethostname(), file=log.v4)


def init_backend_engine(*, config_opts: Optional[Dict[str, Any]] = None):
    """
    Selects the backend engine (TensorFlow, PyTorch, Theano, or whatever)
    and does corresponding initialization and preparation.

    This does not initialize the global ``engine`` object yet.
    See :func:`init_engine` for that.

    This also initializes a new config, if it was not initialized yet,
    and allows to update it via ``config_opts``.

    :param config_opts: to update the global config
    """
    global config
    if config is None:
        config = get_global_config(auto_create=True)
    if config_opts:
        config.update(config_opts)
        if "behavior_version" in config_opts:
            # We actually also do this in init_config, but it might have been updated here.
            BehaviorVersion.set(config.int("behavior_version", None))

    if config.value("PYTORCH_CUDA_ALLOC_CONF", None) is not None:
        # Set this very early, before *any* `import torch`, thus also before select_engine.
        # (It would not hurt if this is set for any non-PT engine.)
        if config.typed_value("torch_distributed") is not None:
            # torchrun (torch.distributed.run) will likely already have imported torch in the worker procs
            # before our code runs here, so it is not safe to assign the env var now.
            # Otherwise, you will likely get such an error:
            #   RuntimeError: config[i] == get()->name() INTERNAL ASSERT FAILED
            #   at "../c10/cuda/CUDACachingAllocator.cpp":1058, please report a bug to PyTorch.
            #   Allocator backend parsed at runtime != allocator backend parsed at load time
            print("Warning: PYTORCH_CUDA_ALLOC_CONF does not work with torch_distributed, not used", file=log.v2)
        else:
            value = config.value("PYTORCH_CUDA_ALLOC_CONF", "")
            print(f"Set PYTORCH_CUDA_ALLOC_CONF={value!r}.", file=log.v3)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value

    BackendEngine.select_engine(config=config)
    if BackendEngine.is_tensorflow_selected():
        print("TensorFlow:", util.describe_tensorflow_version(), file=log.v3)
        if util.get_tensorflow_version_tuple()[0] == 0:
            print("Warning: TF <1.0 is not supported and likely broken.", file=log.v2)
        if os.environ.get("TF_DEVICE"):
            print(
                "Devices: Use %s via TF_DEVICE instead of %s."
                % (os.environ.get("TF_DEVICE"), config.opt_typed_value("device")),
                file=log.v4,
            )
            config.set("device", os.environ.get("TF_DEVICE"))
        if config.is_true("use_horovod"):
            import returnn.tf.horovod

            hvd = returnn.tf.horovod.get_ctx(config=config)
            import socket

            if "gpu" in config.value("device", "") or os.environ.get("CUDA_VISIBLE_DEVICES", ""):
                # We assume that we want to use a GPU.
                gpu_opts = config.typed_dict.setdefault("tf_session_opts", {}).setdefault("gpu_options", {})
                assert "visible_device_list" not in gpu_opts
                gpu_opts["visible_device_list"] = str(hvd.local_rank())
                print(
                    "Horovod: Hostname %s, pid %i, using GPU %s."
                    % (socket.gethostname(), os.getpid(), gpu_opts["visible_device_list"]),
                    file=log.v3,
                )
            else:
                if hvd.rank() == 0:  # Don't spam in all ranks.
                    print("Horovod: Not using GPU.", file=log.v3)
            if hvd.rank() == 0:  # Don't spam in all ranks.
                print("Horovod: Reduce type:", hvd.get_reduce_type(), file=log.v3)
        from returnn.tf.util import basic as tf_util

        tf_session_opts = config.typed_value("tf_session_opts", {})
        assert isinstance(tf_session_opts, dict)
        # This must be done after the Horovod logic, such that we only touch the devices we are supposed to touch.
        tf_util.setup_tf_thread_pools(log_file=log.v3, tf_session_opts=tf_session_opts)
        # Print available devices. Also make sure that get_tf_list_local_devices uses the correct TF session opts.
        tf_util.print_available_devices(tf_session_opts=tf_session_opts, file=log.v2)
        from returnn.tf.native_op import OpMaker

        OpMaker.log_stream = log.v3
        tf_util.debug_register_better_repr()
        if config.is_true("distributed_tf"):
            import returnn.tf.distributed

            returnn.tf.distributed.init_distributed_tf(config)

    elif BackendEngine.is_torch_selected():
        print("PyTorch:", util.describe_torch_version(), file=log.v3)

        if config.typed_value("torch_distributed") is not None:
            import socket
            import returnn.torch.distributed

            torch_distributed = returnn.torch.distributed.get_ctx(config=config)
            print(
                "Torch: Hostname %s, pid %i, using GPU %s."
                % (socket.gethostname(), os.getpid(), str(torch_distributed.local_rank())),
                file=log.v3,
            )

        from returnn.torch.util import diagnose_gpu

        diagnose_gpu.print_relevant_env_vars(file=log.v2)
        diagnose_gpu.print_available_devices(file=log.v2)

        if config.is_true("use_lovely_tensors"):
            try:
                # noinspection PyUnresolvedReferences,PyPackageRequirements
                import lovely_tensors

                lovely_tensors.monkey_patch()
            except ImportError as exc:
                print("Warning: could not import lovely_tensors:", exc, file=log.v3)

    else:
        raise NotImplementedError(f"Backend engine {BackendEngine.get_selected_engine()} not implemented")


def init(config_filename=None, command_line_options=(), config_updates=None, extra_greeting=None):
    """
    :param str|None config_filename:
    :param tuple[str]|list[str]|None command_line_options: e.g. sys.argv[1:]
    :param dict[str]|None config_updates: see :func:`init_config`
    :param str|None extra_greeting:
    """
    debug_util.init_better_exchook()
    util.init_thread_join_hack()
    init_config(
        config_filename=config_filename, command_line_options=command_line_options, extra_updates=config_updates
    )
    if config.bool("use_train_proc_manager", False):
        from returnn.util.train_proc_manager import maybe_start_train_proc_manager

        maybe_start_train_proc_manager(config=config)
    if config.bool("patch_atfork", False):
        from returnn.util.basic import maybe_restart_returnn_with_atfork_patch

        maybe_restart_returnn_with_atfork_patch()
    init_log()
    if extra_greeting:
        print(extra_greeting, file=log.v1)
    returnn_greeting(config_filename=config_filename, command_line_options=command_line_options)
    debug_util.init_faulthandler()
    if config.bool("watch_memory", False):
        from returnn.util.watch_memory import watch_memory

        watch_memory()
    init_backend_engine()
    if config.bool("ipython", False):
        debug_util.init_ipython_kernel()
    if config.typed_value("startup_callback"):
        startup_callback = config.typed_value("startup_callback")
        startup_callback(config=config)
    if need_data():
        if config.bool("use_dummy_datasets", False):
            setup_dummy_datasets()
        init_data()
    print_task_properties()
    init_engine()


def finalize(error_occurred=False):
    """
    Cleanup at the end.

    :param bool error_occurred:
    """
    print("Quitting", file=getattr(log, "v4", sys.stderr))
    global quit_returnn
    quit_returnn = True
    sys.exited = True
    if engine:
        if BackendEngine.is_tensorflow_selected():
            engine.finalize(error_occurred=error_occurred)
            if config.is_true("use_horovod"):
                import horovod.tensorflow as hvd  # noqa

                hvd.shutdown()
        elif BackendEngine.is_torch_selected():
            if config.typed_value("torch_distributed") is not None:
                from torch.distributed import destroy_process_group

                destroy_process_group()


def need_data():
    """
    :return: whether we need to init the data (call :func:`init_data`) for the current task (:func:`execute_main_task`)
    :rtype: bool
    """
    if config.has("need_data") and not config.bool("need_data", True):
        return False
    task = config.value("task", "train")
    if task in ("nop", "nop_init_net_train", "cleanup_old_models"):
        return False
    return True


def execute_main_task():
    """
    Executes the main task (via config ``task`` option).
    """
    from returnn.util.basic import hms_fraction

    start_time = time.time()
    task = config.value("task", "train")
    if config.is_true("dry_run"):
        print("Dry run, will not save anything.", file=log.v1)
    if task == "train":
        # Avoid too many investigations on train_data (like have_seqs()),
        # to avoid triggering any lazy init, which could be unnecessary in the main proc.
        assert train_data, "no train files specified, check 'train' option: %s" % config.value("train", None)
        engine.init_train_from_config(config, train_data, dev_data, eval_data)
        engine.train()
    elif task == "eval":
        if config.value("load", None):
            # this would directly load whatever model is specified
            print("Evaluate model", config.value("load", None), file=log.v2)
            lr_control_update_scores = False
        else:
            # Assume the configured model with some given epoch.
            epoch = config.int("epoch", -1)
            load_epoch = config.int("load_epoch", -1)
            if epoch >= 0:
                assert (load_epoch < 0) or (load_epoch == epoch), "epoch and load_epoch have to match"
                engine.epoch = epoch
                config.set("load_epoch", engine.epoch)
            else:
                assert load_epoch >= 0, "specify epoch or load_epoch"
                engine.epoch = load_epoch
            print("Evaluate epoch", engine.epoch, file=log.v2)
            lr_control_update_scores = True
        engine.init_train_from_config(config, train_data, dev_data, eval_data)
        engine.eval_model(
            output_file=config.value("eval_output_file", None),
            output_per_seq_file=config.value("eval_output_file_per_seq", None),
            loss_name=config.value("loss_name", None),
            output_per_seq_format=config.list("output_per_seq_format", ["score"]),
            output_per_seq_file_format=config.value("output_per_seq_file_format", "txt"),
            lr_control_update_scores=lr_control_update_scores,
        )
    elif task in ["forward", "hpx"]:
        if config.typed_value("forward_callback") or not BackendEngine.is_tensorflow_selected():
            engine.init_network_from_config(config)
            if config.value("forward_data", "eval") in ["train", "dev", "eval"]:
                data = {"train": train_data, "dev": dev_data, "eval": eval_data}[config.value("forward_data", "eval")]
                assert data, "set forward_data"
            else:
                data = init_dataset(config.opt_typed_value("forward_data"))
            # engine.epoch is usually the epoch of the loaded checkpoint,
            # or what EngineBase.get_epoch_model will return.
            # You can have both load and load_epoch, where load points to the checkpoint,
            # and load_epoch is some other epoch, which you will get here for the dataset.
            data.init_seq_order(epoch=engine.epoch or 1)
            forward_callback = config.typed_value("forward_callback")
            assert forward_callback, "no forward_callback specified"
            if callable(forward_callback):
                forward_callback = forward_callback()
            engine.forward_with_callback(dataset=data, callback=forward_callback)
        else:
            assert BackendEngine.is_tensorflow_selected()
            assert eval_data is not None, "no eval data provided"
            combine_labels = config.value("combine_labels", "")
            engine.use_search_flag = config.bool("forward_use_search", False)
            if config.has("epoch"):
                config.set("load_epoch", config.int("epoch", 0))
            engine.init_network_from_config(config)
            eval_data.init_seq_order(epoch=engine.epoch or 1)
            output_file = config.value("output_file", "dump-fwd-epoch-%i.hdf" % engine.epoch)
            forward_batch_size = config.int("forward_batch_size", 0)
            if not forward_batch_size:
                raise Exception("forward_batch_size not set")
            engine.forward_to_hdf(
                data=eval_data,
                output_file=output_file,
                combine_labels=combine_labels,
                batch_size=forward_batch_size,
            )
    elif task == "search":
        engine.use_search_flag = True
        engine.use_eval_flag = config.bool("search_do_eval", True)
        engine.init_network_from_config(config)
        if config.value("search_data", "eval") in ["train", "dev", "eval"]:
            data = {"train": train_data, "dev": dev_data, "eval": eval_data}[config.value("search_data", "eval")]
            assert data, "set search_data"
        else:
            data = init_dataset(config.opt_typed_value("search_data"))
        engine.search(
            data,
            do_eval=config.bool("search_do_eval", True),
            output_layer_names=config.typed_value("search_output_layer", "output"),
            output_file=config.value("search_output_file", ""),
            output_file_format=config.value("search_output_file_format", "txt"),
        )
    elif task == "compute_priors":
        assert train_data is not None, "train data for priors should be provided"
        engine.init_network_from_config(config)
        engine.compute_priors(dataset=train_data, config=config)
    elif task == "analyze":  # anything based on the network + Device
        statistics = config.list("statistics", None)
        engine.init_network_from_config(config)
        engine.analyze(data=eval_data or dev_data, statistics=statistics)
    elif task == "analyze_data":  # anything just based on the data
        analyze_data(config)
    elif task == "hyper_param_tuning":
        import returnn.tf.hyper_param_tuning

        tuner = returnn.tf.hyper_param_tuning.Optimization(config=config, train_data=train_data)
        tuner.work()
    elif task == "cleanup_old_models":
        engine.cleanup_old_models(ask_for_confirmation=True)
    elif task == "search_server":
        engine.use_search_flag = True
        engine.init_network_from_config(config)
        engine.web_server(port=config.int("web_server_port", 12380))
    elif task.startswith("config:"):
        action = config.typed_dict[task[len("config:") :]]
        print("Task: %r" % action, file=log.v1)
        assert callable(action)
        action()
    elif task.startswith("optional-config:"):
        action = config.typed_dict.get(task[len("optional-config:") :], None)
        if action is None:
            print("No task found for %r, so just quitting." % task, file=log.v1)
        else:
            print("Task: %r" % action, file=log.v1)
            assert callable(action)
            action()
    elif task == "nop":
        print("Task: No-operation", file=log.v1)
    elif task == "nop_init_net_train":
        print("Task: No-operation, despite initializing the network (for training)", file=log.v1)
        engine.init_train_from_config(config, train_data, dev_data, eval_data)
    elif task == "initialize_model":
        engine.init_train_from_config(config, train_data, dev_data, eval_data)
        engine.save_model(config.value("model", "dummy"))
    elif task == "debug_shell":
        debug_shell(locals(), globals())
    else:
        raise Exception("unknown task: %r" % (task,))

    print(("elapsed: %s" % hms_fraction(time.time() - start_time)), file=log.v3)


# noinspection PyShadowingNames
def analyze_data(config):  # pylint: disable=redefined-outer-name
    """
    :param Config config:
    """
    dss = config.value("analyze_dataset", "train")
    ds = {"train": train_data, "dev": dev_data, "eval": eval_data}[dss]
    epoch = config.int("epoch", 1)
    print("Analyze dataset", dss, "epoch", epoch, file=log.v1)
    ds.init_seq_order(epoch=epoch)
    stat_prefix = config.value("statistics_save_prefix", "statistics")
    dtype = config.value("statistics_dtype", "float64")
    target = config.value("target", "classes")
    data_key = config.value("data_key", "data")
    assert ds.is_data_sparse(target), "need for prior calculation"
    assert not ds.is_data_sparse(data_key), "needed for mean/var estimation"
    from returnn.util.basic import inplace_increment, progress_bar_with_time, NumbersDict

    priors = numpy.zeros((ds.get_data_dim(target),), dtype=dtype)
    mean = numpy.zeros((ds.get_data_dim(data_key),), dtype=dtype)
    mean_sq = numpy.zeros((ds.get_data_dim(data_key),), dtype=dtype)
    total_targets_len = 0
    total_data_len = 0

    # Note: This is not stable! See :class:`Util.Stats` for a better alternative.
    seq_idx = 0
    while ds.is_less_than_num_seqs(seq_idx):
        progress_bar_with_time(ds.get_complete_frac(seq_idx))
        ds.load_seqs(seq_idx, seq_idx + 1)
        targets = ds.get_data(seq_idx, target)
        inplace_increment(priors, targets, 1)
        total_targets_len += targets.shape[0]
        data = ds.get_data(seq_idx, data_key)
        new_total_data_len = total_data_len + data.shape[0]
        f = float(total_data_len) / new_total_data_len
        mean = mean * f + numpy.sum(data, axis=0) * (1.0 - f)
        mean_sq = mean_sq * f + numpy.sum(data * data, axis=0) * (1.0 - f)
        total_data_len = new_total_data_len
        seq_idx += 1
    log_priors = numpy.log(priors)
    log_priors -= numpy.log(NumbersDict(ds.get_num_timesteps())[target])
    std_dev = numpy.sqrt(mean_sq - mean * mean)
    print("Finished. %i total target frames, %i total data frames" % (total_targets_len, total_data_len), file=log.v1)
    priors_fn = stat_prefix + ".log_priors.txt"
    mean_fn = stat_prefix + ".mean.txt"
    std_dev_fn = stat_prefix + ".std_dev.txt"
    print("Dump priors to", priors_fn, file=log.v1)
    numpy.savetxt(priors_fn, log_priors)
    print("Dump mean to", mean_fn, file=log.v1)
    numpy.savetxt(mean_fn, mean)
    print("Dump std dev to", std_dev_fn, file=log.v1)
    numpy.savetxt(std_dev_fn, std_dev)
    print("Done.", file=log.v1)


def main(argv=None):
    """
    Main entry point of RETURNN.

    :param list[str]|None argv: ``sys.argv`` by default
    """
    if argv is None:
        argv = sys.argv
    return_code = 0
    try:
        assert len(argv) >= 2, "usage: %s <config>" % argv[0]
        init(command_line_options=argv[1:])
        execute_main_task()
    except KeyboardInterrupt:
        return_code = 1
        print("KeyboardInterrupt", file=getattr(log, "v3", None) or sys.stderr)
        if getattr(log, "verbose", None) and log.verbose[5]:
            sys.excepthook(*sys.exc_info())
    finalize(error_occurred=return_code != 0)
    if return_code:
        sys.exit(return_code)


if __name__ == "__main__":
    main(sys.argv)
