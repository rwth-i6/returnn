"""
Provides :class:`Config` and some related helpers.
"""

from __future__ import annotations

__author__ = "Patrick Doetsch"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]

from typing import Optional, Any, Dict, List
import contextlib
import sys
import os
import types as _types


class Config:
    """
    Reads in some config file, and provides access to the key/value items.
    We support some simple text-line-based config, JSON, and Python format.
    """

    def __init__(self, items: Optional[Dict[str, Any]] = None):
        """
        :param items: optional initial typed_dict
        """
        self.dict: Dict[str, List[str]] = {}
        self.typed_dict: Dict[str, Any] = {}  # could be loaded via JSON or so
        self.network_topology_json: Optional[str] = None
        self.files = []
        if items is not None:
            self.typed_dict.update(items)

    def __getstate__(self):
        import io
        from returnn.util.task_system import Pickler

        class _CustomPickler(Pickler):
            use_whichmodule = False

            # This will trigger potentially some fallback of task_system.Pickler,
            # like _save_type_fallback.
            # Also, do not use the intelligent logic for our own dict.
            # We explicitly want to pickle it as-is.
            module_name_black_list = {_PyModuleName}

        buffer = io.BytesIO()
        pickler = _CustomPickler(buffer)
        memo_idx = len(pickler.memo)
        pickler.memo[id(self)] = memo_idx, self
        pickler.dump(self.typed_dict)

        return {
            "_pid": os.getpid(),
            "_self_memo_idx": memo_idx,
            "_typed_dict_pickled": buffer.getvalue(),
            "_is_global": self is get_global_config(raise_exception=False),
        }

    def __setstate__(self, state):
        import io

        # Use pure-Python unpickling to be able to extend the memo.
        # noinspection PyUnresolvedReferences,PyProtectedMember
        from pickle import _Unpickler

        self.__init__()

        buffer = io.BytesIO(state["_typed_dict_pickled"])
        unpickler = _Unpickler(buffer)
        unpickler.memo[state["_self_memo_idx"]] = self
        self.typed_dict = unpickler.load()

        if state["_is_global"] and os.getpid() != state["_pid"]:
            set_global_config(self)
            _global_config_as_py_module_proxy_setup()

    def load_file(self, f):
        """
        Reads the configuration parameters from a file and adds them to the inner set of parameters.

        :param string|io.TextIOBase|io.StringIO f:
        """
        if isinstance(f, str):
            assert os.path.isfile(f), "config file not found: %r" % f
            self.files.append(f)
            filename = f
            dirname = os.path.dirname(filename) or "."
            content = open(filename).read()
        else:
            # assume stream-like
            filename = "<config string>"
            dirname = None
            content = f.read()
        content = content.strip()
        if content.startswith("#!") or filename.endswith(".py"):  # assume Python
            if dirname and os.path.exists(f"{dirname}/__init__.py") and filename.endswith(".py"):
                # It looks like a Python module inside a Python package.
                # Import it as a module.
                import importlib

                basedir = os.path.abspath(dirname)
                while os.path.exists(f"{basedir}/__init__.py"):
                    basedir = os.path.dirname(basedir)
                if basedir not in sys.path:
                    sys.path.insert(0, basedir)
                modname = os.path.relpath(dirname, basedir).replace("/", ".") + "." + os.path.basename(filename)[:-3]
                mod = importlib.import_module(modname)
                self.update(vars(mod))

            else:
                # Directly execute the Python code.
                from returnn.util.basic import custom_exec

                # Operate inplace on ourselves.
                # Also, we want that it's available as the globals() dict, so that defined functions behave well
                # (they would lose the local context otherwise).
                user_ns = self.typed_dict
                # Always overwrite:
                user_ns.update({"config": self, "__file__": filename, "__name__": _PyModuleName})
                custom_exec(content, filename, user_ns, user_ns)
                _global_config_as_py_module_proxy_setup()
            return
        if content.startswith("{"):  # assume JSON
            from returnn.util.basic import load_json

            json_content = load_json(content=content)
            assert isinstance(json_content, dict)
            self.update(json_content)
            return
        # old line-based format
        for line in content.splitlines():
            if "#" in line:  # Strip away comment.
                line = line[: line.index("#")]
            line = line.strip()
            if not line:
                continue
            line = line.split(None, 1)
            assert len(line) == 2, "unable to parse config line: %r" % line
            self.add_line(key=line[0], value=line[1])

    @classmethod
    def get_config_file_type(cls, f):
        """
        :param str f: file path
        :return: "py", "js" or "txt"
        :rtype: str
        """
        with open(f, "r") as f:
            start = f.read(3)
        if start.startswith("#!"):
            return "py"
        if start.startswith("{"):
            return "js"
        return "txt"

    def parse_cmd_args(self, args):
        """
        :param list[str]|tuple[str] args:
        """
        from optparse import OptionParser

        parser = OptionParser()
        parser.add_option(
            "-a",
            "--activation",
            dest="activation",
            help="[STRING/LIST] Activation functions: logistic, tanh, softsign, relu, identity, zero, one, maxout.",
        )
        parser.add_option(
            "-b",
            "--batch_size",
            dest="batch_size",
            help="[INTEGER/TUPLE] Maximal number of frames per batch (optional: shift of batching window).",
        )
        parser.add_option(
            "-c",
            "--chunking",
            dest="chunking",
            help="[INTEGER/TUPLE] Maximal number of frames per sequence (optional: shift of chunking window).",
        )
        parser.add_option("-d", "--description", dest="description", help="[STRING] Description of experiment.")
        parser.add_option("-e", "--epoch", dest="epoch", help="[INTEGER] Starting epoch.")
        parser.add_option("-E", "--eval", dest="eval", help="[STRING] eval file path")
        parser.add_option(
            "-f",
            "--gate_factors",
            dest="gate_factors",
            help="[none/local/global] Enables pooled (local) or separate (global) coefficients on gates.",
        )
        parser.add_option("-g", "--lreg", dest="lreg", help="[FLOAT] L1 or L2 regularization.")
        parser.add_option(
            "-i",
            "--save_interval",
            dest="save_interval",
            help="[INTEGER] Number of epochs until a new model will be saved.",
        )
        parser.add_option("-j", "--dropout", dest="dropout", help="[FLOAT] Dropout probability (0 to disable).")
        parser.add_option(
            "-k", "--output_file", dest="output_file", help="[STRING] Path to target file for network output."
        )
        parser.add_option("-l", "--log", dest="log", help="[STRING] Log file path.")
        parser.add_option("-L", "--load", dest="load", help="[STRING] load model file path.")
        parser.add_option(
            "-m", "--momentum", dest="momentum", help="[FLOAT] Momentum term in gradient descent optimization."
        )
        parser.add_option(
            "-n", "--num_epochs", dest="num_epochs", help="[INTEGER] Number of epochs that should be trained."
        )
        parser.add_option("-o", "--order", dest="order", help="[default/sorted/random] Ordering of sequences.")
        parser.add_option("-p", "--loss", dest="loss", help="[loglik/sse/ctc] Objective function to be optimized.")
        parser.add_option(
            "-q",
            "--cache",
            dest="cache",
            help="[INTEGER] Cache size in bytes (supports notation for kilo (K), mega (M) and gigabyte (G)).",
        )
        parser.add_option(
            "-r",
            "--learning_rate",
            dest="learning_rate",
            help="[FLOAT] Learning rate in gradient descent optimization.",
        )
        parser.add_option(
            "-s", "--hidden_sizes", dest="hidden_sizes", help="[INTEGER/LIST] Number of units in hidden layers."
        )
        parser.add_option(
            "-t",
            "--truncate",
            dest="truncate",
            help="[INTEGER] Truncates sequence in BPTT routine after specified number of timesteps (-1 to disable).",
        )
        parser.add_option(
            "-u",
            "--device",
            dest="device",
            help="[STRING/LIST] CPU and GPU devices that should be used (example: gpu0,cpu[1-6] or gpu,cpu*).",
        )
        parser.add_option("-v", "--verbose", dest="log_verbosity", help="[INTEGER] Verbosity level from 0 - 5.")
        parser.add_option("-w", "--window", dest="window", help="[INTEGER] Width of sliding window over sequence.")
        parser.add_option("-x", "--task", dest="task", help="[train/forward/analyze] Task of the current program call.")
        parser.add_option(
            "-y", "--hidden_type", dest="hidden_type", help="[VALUE/LIST] Hidden layer types: forward, recurrent, lstm."
        )
        parser.add_option(
            "-z", "--max_sequences", dest="max_seqs", help="[INTEGER] Maximal number of sequences per batch."
        )
        parser.add_option("--config", dest="load_config", help="[STRING] load config")
        (options, args) = parser.parse_args(list(args))
        options = vars(options)
        for opt in options.keys():
            if options[opt] is not None:
                if opt == "load_config":
                    self.load_file(options[opt])
                else:
                    self.add_line(opt, options[opt])
        assert len(args) % 2 == 0, "expect (++key, value) config tuples in remaining args: %r" % args
        for i in range(0, len(args), 2):
            key, value = args[i : i + 2]
            assert key[0:2] == "++", "expect key prefixed with '++' in (%r, %r)" % (key, value)
            if value[:2] == "+-":
                value = value[1:]  # otherwise we never could specify things like "++threshold -0.1"
            self.add_line(key=key[2:], value=value)

    def add_line(self, key, value):
        """
        Adds one specific configuration (key,value) pair to the inner set of parameters
        :type key: str
        :type value: str
        """
        if key in self.typed_dict:
            # This is a special case. We overwrite a config value which was typed before.
            # E.g. this could have been loaded via a Python config file.
            # We want to keep the entry in self.typed_dict because there might be functions/lambdas inside
            # the config which require the global variable to be available.
            # See :func:`test_rnn_init_config_py_global_var`.
            value_type = type(self.typed_dict[key])
            if value_type == str:
                pass  # keep as-is
            else:
                try:
                    value = eval(value)
                except SyntaxError:
                    from returnn.log import log

                    print(
                        "WARNING: can't evaluate config param %r to previous type: %s. Keeping as string."
                        % (value, value_type),
                        file=log.v1,
                    )
            self.typed_dict[key] = value
            return
        if value.find(",") > 0:
            value = value.split(",")
        else:
            value = [value]
        if key == "include":
            for f in value:
                self.load_file(f)
        else:
            self.dict[key] = value

    def has(self, key):
        """
        Returns whether the given key is present in the inner set of parameters
        :type key: string
        :rtype: boolean
        :returns True if and only if the given key is in the inner set of parameters
        """
        if key in self.typed_dict:
            return True
        return key in self.dict

    def is_typed(self, key):
        """
        :type key: string
        :rtype: boolean
        :returns True if and only if the value of the given key has a specified data type
        """
        return key in self.typed_dict

    def is_true(self, key, default=False):
        """
        :param str key:
        :param bool default:
        :return: bool(value) if it is set or default
        :rtype: bool
        """
        if self.is_typed(key):
            return bool(self.typed_dict[key])
        return self.bool(key, default=default)

    def is_of_type(self, key, types):
        """
        :param str key:
        :param type|tuple[type] types: for isinstance() check
        :return: whether is_typed(key) is True and isinstance(value, types) is True
        :rtype: bool
        """
        if key in self.typed_dict:
            return isinstance(self.typed_dict[key], types)
        return False

    def get_of_type(self, key, types, default=None):
        """
        :param str key:
        :param type|list[type]|T types: for isinstance() check
        :param T|None default:
        :return: if is_of_type(key, types) is True, returns the value, otherwise default
        :rtype: T
        """
        if self.is_of_type(key, types):
            return self.typed_dict[key]
        return default

    def set(self, key, value):
        """
        :type key: str
        :type value: list[str] | str | int | float | bool | dict | None
        """
        self.typed_dict[key] = value

    def update(self, dikt):
        """
        :type dikt: dict
        """
        for key, value in dikt.items():
            self.set(key, value)

    def _hack_value_reading_debug(self):
        orig_value_func = self.value

        def wrapped_value_func(*args, **kwargs):
            """
            Wrapped func.
            """
            res = orig_value_func(*args, **kwargs)
            print(
                "Config.value(%s) -> %r"
                % (", ".join(list(map(repr, args)) + ["%s=%r" % (k, v) for (k, v) in kwargs.items()]), res)
            )
            return res

        setattr(self, "value", wrapped_value_func)

    def value(self, key, default, index=None, list_join_str=","):
        """
        :type key: str
        :type default: T
        :type index: int | None
        :param str list_join_str:
        :rtype: str | T
        """
        if key in self.typed_dict:
            ls = self.typed_dict[key]
            if index is None:
                if isinstance(ls, (list, tuple)):
                    return list_join_str.join([str(v) for v in ls])
                elif ls is None:
                    return default
                else:
                    return str(ls)
            else:
                return str(ls[index])
        if key in self.dict:
            ls = self.dict[key]
            if index is None:
                return list_join_str.join(ls)
            else:
                return ls[index]
        return default

    def typed_value(self, key, default=None, index=None):
        """
        :type key: str
        :type default: T
        :type index: int | None
        :rtype: T | typing.Any
        """
        value = self.typed_dict.get(key, default)
        if index is not None:
            assert isinstance(index, int)
            if isinstance(value, (list, tuple)):
                value = value[index]
            else:
                assert index == 0
        return value

    def opt_typed_value(self, key, default=None):
        """
        :param str key:
        :param T|None default:
        :rtype: T|object|str|None
        """
        if key in self.typed_dict:
            return self.typed_dict[key]
        return self.value(key, default)

    def int(self, key, default, index=0):
        """
        Parses the value of the given key as integer, returning default if not existent
        :type key: str
        :type default: T
        :type index: int
        :rtype: int | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
            if value is not None:
                assert isinstance(value, int)
            return value
        if key in self.dict:
            return int(self.value(key, default, index))
        return default

    def bool(self, key, default, index=0):
        """
        Parses the value of the given key as boolean, returning default if not existent
        :type key: str
        :type default: T
        :type index: int
        :rtype: bool | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
            if isinstance(value, int):
                value = bool(value)
            if value is not None:
                assert isinstance(value, bool)
            return value
        if key not in self.dict:
            return default
        v = str(self.value(key, None, index))
        if not v:
            return default
        from returnn.util.basic import to_bool

        return to_bool(v)

    def bool_or_other(self, key, default=None, index=0):
        """
        :param str key:
        :param T default:
        :param int index:
        :return: if we have typed value, just as-is. otherwise try to convert to bool. or default if not there.
        :rtype: bool|T|object
        """
        if key in self.typed_dict:
            return self.typed_value(key, default=default, index=index)
        if key not in self.dict:
            return default
        v = str(self.value(key, None, index))
        if not v:
            return default
        from returnn.util.basic import to_bool

        try:
            return to_bool(v)
        except ValueError:
            return v

    def float(self, key, default, index=0):
        """
        Parses the value of the given key as float, returning default if not existent
        :type key: str
        :type default: T
        :type index: int
        :rtype: float | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
        else:
            value = self.value(key, default, index)
        if value is not None:
            if isinstance(value, str):
                # Special case for float as str. We automatically cast this case.
                # This is also to handle special values such as "inf".
                value = float(value)
            assert isinstance(value, (int, float))
        return value

    def list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[str] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            return list(value)
        if key not in self.dict:
            return default
        return self.dict[key]

    def int_list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[int] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            for x in value:
                assert isinstance(x, int)
            return list(value)
        return [int(x) for x in self.list(key, default)]

    def float_list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[float] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            for x in value:
                assert isinstance(x, (float, int))
            return list(value)
        return [float(x) for x in self.list(key, default)]

    def int_pair(self, key, default=None):
        """
        :param str key:
        :param (int,int)|None default:
        :rtype: (int,int)
        """
        if default is None:
            default = (0, 0)
        if not self.has(key):
            return default
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if not isinstance(value, (tuple, list)):
                value = (value, value)
            assert len(value) == 2
            for x in value:
                assert isinstance(x, int)
            return tuple(value)
        value = self.value(key, "")
        if ":" in value:
            return int(value.split(":")[0]), int(value.split(":")[1])
        else:
            return int(value), int(value)


_global_config: Optional[Config] = None


@contextlib.contextmanager
def global_config_ctx(config: Optional[Config]):
    """
    sets the config as global config in this context,
    and recovers the original global config afterwards
    """
    global _global_config
    prev_global_config = _global_config
    try:
        set_global_config(config)
        yield
    finally:
        _global_config = prev_global_config


def set_global_config(config: Optional[Config]):
    """
    Will define the global config, returned by :func:`get_global_config`

    :param config:
    """
    _get_or_set_config_via_tf_default_graph(config, assign=True)
    global _global_config
    _global_config = config


def get_global_config(*, raise_exception: bool = True, auto_create: bool = False, return_empty_if_none: bool = False):
    """
    :param raise_exception: if no global config is found, raise an exception, otherwise return None
    :param auto_create: if no global config is found, it creates one, registers it as global, and returns it
    :param return_empty_if_none: if no global config is found, it creates one (which is empty) and returns it
    :rtype: Config|None
    """
    config = _get_or_set_config_via_tf_default_graph()
    if config:
        return config
    if _global_config:
        return _global_config
    # We are the main process.
    import sys

    main_mod = sys.modules["__main__"]  # should be rnn.py
    if hasattr(main_mod, "config") and isinstance(main_mod.config, Config):
        return main_mod.config
    # Maybe __main__ is not rnn.py, or config not yet loaded.
    # Anyway, try directly. (E.g. for SprintInterface.)
    import returnn.__main__ as rnn

    if isinstance(rnn.config, Config):
        return rnn.config
    if auto_create:
        config = Config()
        set_global_config(config)
        return config
    if return_empty_if_none:
        return Config()
    if raise_exception:
        raise Exception("No global config found.")
    return None


def _get_or_set_config_via_tf_default_graph(config: Optional[Config] = None, assign: bool = False):
    """
    This is done in a safe way, and might just be a no-op.
    When TF is not imported yet, it will just return.

    :param config: if set, will set it
    :param assign: whether to set it (even if config=None)
    :rtype: Config|None
    """
    if "tensorflow" not in sys.modules:
        return None
    from returnn.tf.compat import v1 as tf_v1

    graph = tf_v1.get_default_graph()
    # We could use collection refs, but this could cause other problems,
    # and is more complicated than what we need.
    # We just use a custom own attrib.
    attrib_name = "_RETURNN_config_in_graph"
    if assign or config:
        setattr(graph, attrib_name, config)
    return getattr(graph, attrib_name, None)


def network_json_from_config(config):
    """
    :param Config config:
    :rtype: dict[str]
    """
    if config.has("network") and config.is_typed("network"):
        json_content = config.typed_value("network")
        assert isinstance(json_content, dict)
        assert json_content
        return json_content
    else:
        raise ValueError("Network is not defined in config. Define `network`.")


def tf_should_use_gpu(config):
    """
    :param Config config:
    :rtype: bool
    """
    cfg_dev = config.value("device", None)
    # Short path.
    if cfg_dev == "gpu":
        return True
    if cfg_dev == "cpu":
        return False
    if not cfg_dev:
        # Better default: Use GPU if available.
        from returnn.log import log
        from returnn.tf.util.basic import is_gpu_available

        if is_gpu_available():
            print("Device not set explicitly, and we found a GPU, which we will use.", file=log.v2)
            config.set("device", "gpu")
            return True
        else:
            print("Device not set explicitly, and no GPU found.", file=log.v2)
            config.set("device", "cpu")
            return False
    else:
        raise ValueError("Currently unsupported TF device %r specified" % (cfg_dev,))


_PyModuleName = "__returnn_config__"


def _global_config_as_py_module_proxy_setup():
    if _PyModuleName in sys.modules:
        return
    sys.modules[_PyModuleName] = _GlobalConfigAsPyModuleProxy(_PyModuleName)


class _GlobalConfigAsPyModuleProxy(_types.ModuleType):
    """
    Takes :func:`get_global_config`, and makes its ``typed_dict`` available as module attributes.
    """

    @staticmethod
    def _get_config() -> Optional[Config]:
        """
        :return: config or None if not available anymore
        """
        return get_global_config(raise_exception=False)

    def __getattribute__(self, item):
        # Implement also __getattribute__ such that early access to just self.__dict__ (e.g. via vars(self)) also works.
        if item == "__dict__":
            cfg: Optional[Config] = self._get_config()
            if not cfg:
                return {}
            return cfg
        return super().__getattribute__(item)

    def __getattr__(self, item):
        cfg: Optional[Config] = self._get_config()
        if not cfg:
            raise AttributeError("config %s not loaded anymore" % self.__name__)
        if item not in cfg.typed_dict:
            raise AttributeError("config %s has no attribute %r" % (self.__name__, item))
        return cfg.typed_dict[item]

    def __dir__(self):
        cfg: Optional[Config] = self._get_config()
        if not cfg:
            return []
        return sorted(cfg.typed_dict.keys())

    def __setattr__(self, key, value):
        if key in ["__file__"]:
            super().__setattr__(key, value)
            return
        cfg: Optional[Config] = self._get_config()
        if cfg:
            cfg.typed_dict[key] = value


class SubProcCopyGlobalConfigPreInitFunc:
    """
    A pre-init hook for a subprocess which copies the global config to the subprocess.

    It can be important that this init function is called even before the unpickling of other data,
    as that unpickling might depend on the right context, e.g. having the global RETURNN config.
    Example issue with MultiProcDataset: https://github.com/rwth-i6/returnn/issues/1495

    Example usage::

        NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())
    """

    def __init__(self):
        # Get the RETURNN global config here. Allow this to be optional (for use outside of RETURNN).
        # We store it here such that pickling this worker init func will also pickle the config,
        # so that we can reset it as global config inside the worker.
        # Some RETURNN datasets might depend on the config.
        # https://github.com/rwth-i6/returnn/issues/1495
        # MultiProcDataset has a similar logic, see https://github.com/rwth-i6/returnn/issues/1384.
        self.global_config = get_global_config(raise_exception=False)

    def __call__(self):
        from returnn.util import better_exchook
        from returnn import __old_mod_loader__

        better_exchook.install()
        __old_mod_loader__.disable_lazy_mod_loads()

        if self.global_config:
            set_global_config(self.global_config)
