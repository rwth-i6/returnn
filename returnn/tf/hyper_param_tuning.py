"""
Here we provide some logic to perform hyper-parameter search.
See ``demos/demo-hyper-param-tuning.config`` for an example config.
For each entry in the config where search should be performed on,
you declare it as an instance of :class:`HyperParam`.
Then, this module will find all such instances in the config and replace it with values during search.

The search itself is some evolutionary genetic search.
There are many variants to it, e.g. such as what kind of manipulations you,
e.g. cross-over and mutation, and also, how you sample new random values.
The current logic probably can be improved.

Currently, each search is a training started from the scratch, and the accumulated train score
is used as an evaluation measure.
This probably also can be improved.
Also, instead of always starting from scratch, we could keep intermediate results and resume from them,
or use real training intermediate results and resume from them.
We could even do some simple search in the beginning of each epoch when we keep it cheap enough.

Also, we could store the population of hyper params on disk to allow resuming of a search.
"""

from __future__ import annotations

import sys
import time
import typing
import numpy
import returnn.tf.compat as tf_compat
from returnn.config import Config
from returnn.log import log
from returnn.datasets import Dataset
from returnn.datasets.generating import StaticDataset
from returnn.tf.engine import Engine, Runner, CancelTrainingException
from returnn.util.basic import CollectionReadCheckCovered, hms_fraction, guess_requested_max_num_threads


Eps = 1e-16


class HyperParam:
    """
    Represents one hyper parameter.
    """

    # noinspection PyShadowingNames
    def __init__(self, dtype=None, bounds=None, classes=None, log=False, default=None):
        """
        :param str|type|None|list dtype: e.g. "float", "int" or "bool", or if Collection, will be classes
        :param None|list[int|float] bounds: inclusive
        :param list|None classes:
        :param bool log: if in log-scale
        :param float|int|object|None default:
        """
        if isinstance(dtype, (list, tuple)):
            assert classes is None
            assert bounds is None
            classes = dtype
            dtype = None
        if dtype is None:
            assert classes is not None
        elif dtype == "float":
            dtype = float
        elif dtype == "int":
            dtype = int
        elif dtype == "bool":
            dtype = bool
        assert dtype in (float, int, bool, None)
        if bounds is not None:
            assert dtype in (int, float)
            assert isinstance(bounds, (list, tuple))
            assert len(bounds) == 2
            assert dtype(bounds[0]) < dtype(bounds[1])
        if classes is not None:
            assert isinstance(classes, (list, tuple)), "should be with a defined order"
            assert len(classes) > 0
        self.dtype = dtype
        self.bounds = bounds
        self.classes = classes
        self.log_space = log
        self.default = default
        self.unique_idx = HyperParam._get_next_unique_idx()
        self.usages = []  # type: typing.List[_AttrChain]

    _unique_idx = 0

    @classmethod
    def _get_next_unique_idx(cls):
        cls._unique_idx += 1
        return cls._unique_idx

    def __repr__(self):
        if self.classes is not None:
            return "HyperParam(%r)" % self.classes
        dtype_name = self.dtype.__name__
        ext = ""
        if self.log_space:
            ext += ", log=True"
        if self.default is not None:
            ext += ", default=%r" % self.default
        if self.bounds is not None:
            return "HyperParam(%s, %s%s)" % (dtype_name, self.bounds, ext)
        assert self.bounds is None
        return "HyperParam(%s%s)" % (dtype_name, ext)

    def get_canonical_usage(self):
        """
        :rtype: _AttrChain
        """
        return self.get_sorted_usages()[0]

    def get_sorted_usages(self):
        """
        :rtype: list[_AttrChain]
        """
        return sorted(self.usages, key=lambda chain: min(2, len(chain.chain)))

    def description(self):
        """
        :rtype: str
        """
        if len(self.usages) == 0:
            usage_str = "<no usage>"
        elif len(self.usages) == 1:
            usage_str = str(self.usages[0])
        else:
            usage_str = str(self.get_canonical_usage()) + "|..."
        return usage_str + ": %s" % self

    def get_num_instances(self, upper_limit=100):
        """
        :param int upper_limit:
        :rtype: int
        """
        assert upper_limit >= 2
        if self.classes is not None:
            return min(len(self.classes), upper_limit)
        if self.dtype is bool:
            return 2
        if self.dtype is float:
            return upper_limit
        if self.dtype is int:
            x1, x2 = self.bounds
            x1 = numpy.ceil(x1)
            x2 = numpy.floor(x2)
            assert x1 < x2
            return min(x2 - x1 + 1, upper_limit)
        raise Exception("invalid dtype %r" % self.dtype)

    def merge_values(self, value1, value2):
        """
        Merge two values, which are valid values for this `HyperParam`.

        :param T value1:
        :param T value2:
        :rtype: T
        """
        if self.dtype is bool:
            return value1
        if self.log_space:
            x0, x1 = value1, value2
            if x0 > x1:
                x0, x1 = x1, x0
            if x0 < 0 or x1 < 0:
                assert x0 <= x1 <= 0
                sign = -1
                x0, x1 = -x1, -x0
            else:
                sign = 1
            assert x1 >= x0 >= 0
            x0o = x0
            if x0 < Eps * 0.5:
                x0 = Eps * 0.5
            if x1 < Eps:
                x1 = Eps
            x0 = numpy.log(float(x0))
            x1 = numpy.log(float(x1))
            y = numpy.exp(x0 + (x1 - x0) * 0.5)
            if y <= Eps:
                y = x0o
            return self.dtype(y) * sign
        if self.dtype is int:
            return (value1 + value2) // 2
        return self.dtype((value1 + value2) * 0.5)

    def get_value(self, selected, eps=Eps):
        """
        :param float selected: must be between 0 and 1
        :param float eps: if in log-space and you have e.g. bounds=[0,1], will be the lowest value, before 0. see code.
        :rtype: float|int|bool|object
        """
        assert 0 < eps
        assert 0 <= selected <= 1
        if self.classes:
            return self.classes[int(len(self.classes) * selected)]
        if self.dtype is bool:
            return selected > 0.5
        if self.bounds:
            if self.dtype is int and not self.log_space:
                return self.bounds[0] + int((self.bounds[1] - self.bounds[0]) * selected)
            if self.log_space:
                x0, x1 = self.bounds
                if x0 < 0 or x1 < 0:
                    assert x0 < x1 <= 0
                    sign = -1
                    x0, x1 = -x1, -x0
                else:
                    sign = 1
                assert x1 > x0 >= 0
                x0b, x1b = x0, x1
                if x0b < eps * 0.5:
                    x0b = eps * 0.5
                if x1b < eps:
                    x1b = eps
                x0l = numpy.log(float(x0b))
                x1l = numpy.log(float(x1b))
                y = numpy.exp(x0l + (x1l - x0l) * selected)
                if y <= eps:
                    y = x0
                return self.dtype(y) * sign
            return self.dtype(self.bounds[0] + (self.bounds[1] - self.bounds[0]) * selected)
        # No bounds. So anything -inf to inf.
        # But exclude -inf/inf.
        # Assume selected is uniform in [0,1], so use the inverse accumulated Gauss density function
        # to get normal distributed in [-inf,inf].
        x = selected
        if x < eps:
            x = eps
        if x > 1.0 - eps:
            x = 1.0 - eps
        import scipy.special

        return self.dtype(scipy.special.ndtri(x))

    def get_initial_value(self):
        """
        :rtype: float|int|bool|object
        """
        return self.get_value(selected=0.5)

    def get_default_value(self):
        """
        :rtype: float|int|bool|object
        """
        if self.default is not None:
            return self.dtype(self.default)
        return self.get_initial_value()

    def get_random_value(self, seed, eps=Eps):
        """
        :param int seed:
        :param float eps: see get_value()
        :rtype: float|int|bool|object
        """
        rnd = numpy.random.RandomState(seed=seed)
        x = rnd.uniform(0.0, 1.0)
        if x < eps:
            x = 0.0
        if x > 1.0 - eps:
            x = 1.0
        return self.get_value(x, eps=eps)

    def get_random_value_by_idx(self, iteration_idx, individual_idx):
        """
        :param int iteration_idx:
        :param int individual_idx:
        :rtype: float|int|bool|object
        """
        # Use a deterministic seed for the random number generator
        # which will not change on unrelated changes in the config file,
        # so that runs will stay deterministic in this sense.
        seed = hash_obj((self.get_canonical_usage(), iteration_idx, individual_idx))
        return self.get_random_value(seed=seed)


class TrainException(Exception):
    """
    Exception from training.
    """


class Individual:
    """
    One instance of hyper params.
    """

    def __init__(self, hyper_param_mapping, name):
        """
        :param dict[HyperParam] hyper_param_mapping:
        :param str name:
        """
        self.hyper_param_mapping = hyper_param_mapping
        self.cost = None
        self.name = name

    def cross_over(self, hyper_params, population, random_seed):
        """
        :param list[HyperParam] hyper_params:
        :param list[Individual] population:
        :param int random_seed:
        :return: copy of self, cross-overd with others
        :rtype: Individual
        """
        name = self.name
        if len(name) > 10:
            name = name[:8] + ".."
        name += "x%x" % random_seed
        res = Individual(hyper_param_mapping=self.hyper_param_mapping.copy(), name=name)
        rnd = numpy.random.RandomState(random_seed)
        while True:
            other = population[rnd.random_integers(0, len(population) - 1)]
            for p in hyper_params:
                x = rnd.uniform(0.0, 1.0)
                if x > 0.75:
                    res.hyper_param_mapping[p] = other.hyper_param_mapping[p]
                elif x > 0.5:
                    res.hyper_param_mapping[p] = p.merge_values(
                        res.hyper_param_mapping[p], other.hyper_param_mapping[p]
                    )
            if rnd.uniform(0.0, 1.0) > 0.5:
                break
        return res


class Optimization:
    """
    Hyper parameter optimization handler class.
    """

    def __init__(self, config, train_data):
        """
        :param returnn.config.Config config:
        :param Dataset train_data:
        """
        self.config = config
        self.opts = CollectionReadCheckCovered(config.get_of_type("hyper_param_tuning", dict, {}))
        self.log = log.v1
        train_data.init_seq_order(epoch=1)
        self.train_data = StaticDataset.copy_from_dataset(train_data, max_seqs=self.opts.get("num_train_steps", 100))
        self.hyper_params = []  # type: typing.List[HyperParam]
        self._find_hyper_params()
        if not self.hyper_params:
            raise Exception("No hyper params found.")
        self.hyper_params.sort(key=lambda p_: p_.unique_idx)
        print("We have found these hyper params:")
        for p in self.hyper_params:
            print(" %s" % p.description())
        self.dry_run_first_individual = self.opts.get("dry_run_first_individual", True)
        self.num_iterations = self.opts["num_tune_iterations"]
        self.num_individuals = self.opts["num_individuals"]
        self.num_kill_individuals = self.opts.get("num_kill_individuals", self.num_individuals // 2)
        self.num_best = self.opts.get("num_best", 10)
        self.num_threads = self.opts.get("num_threads", guess_requested_max_num_threads())
        self.opts.assert_all_read()

    def _find_hyper_params(self, base=None, visited=None):
        """
        :param _AttrChain base:
        :param set[int] visited: set of ids
        """
        from inspect import ismodule

        if base is None:
            base = _AttrChain(base=self.config)
        if isinstance(base.value, HyperParam):
            base.value.usages.append(base)
            if base.value not in self.hyper_params:
                self.hyper_params.append(base.value)
            return
        if visited is None:
            visited = set()
        if id(base.value) in visited:
            return
        visited.add(id(base.value))
        if ismodule(base.value):
            return
        if isinstance(base.value, dict):
            col_type = _AttribOrKey.ColTypeDict
            keys = base.value.keys()
        elif isinstance(base.value, Config):
            col_type = _AttribOrKey.ColTypeConfig
            keys = base.value.typed_dict.keys()
        else:
            # Add other specific object types, but not in generic all.
            return
        for key in sorted(keys):
            child = base.get_extended_chain(_AttribOrKey(key=key, col_type=col_type))
            self._find_hyper_params(base=child, visited=visited)

    def get_population(self, iteration_idx, num_individuals):
        """
        :param int iteration_idx:
        :param int num_individuals:
        :rtype: list[Individual]
        """
        assert num_individuals > 0
        return [self.get_individual(iteration_idx=iteration_idx, individual_idx=i) for i in range(num_individuals)]

    def get_individual(self, iteration_idx, individual_idx):
        """
        :param int iteration_idx:
        :param int individual_idx:
        :rtype: Individual
        """
        return Individual(
            {
                p: p.get_random_value_by_idx(iteration_idx=iteration_idx, individual_idx=individual_idx)
                for p in self.hyper_params
            },
            name="%i-%i" % (iteration_idx, individual_idx),
        )

    def cross_over(self, population, iteration_idx):
        """
        :param list[Individual] population: modified in-place
        :param int iteration_idx:
        """
        for i in range(len(population) - 1):
            population[i] = population[i].cross_over(
                hyper_params=self.hyper_params,
                population=population[:i] + population[i + 1 :],
                random_seed=iteration_idx * 1013 + i * 17,
            )

    def create_config_instance(self, hyper_param_mapping, gpu_ids):
        """
        :param dict[HyperParam] hyper_param_mapping: maps each hyper param to some value
        :param set[int] gpu_ids:
        :rtype: Config
        """
        assert set(self.hyper_params) == set(hyper_param_mapping.keys())
        from returnn.util.basic import deepcopy

        config = deepcopy(self.config)
        assert isinstance(config, Config)
        for p, value in hyper_param_mapping.items():
            assert isinstance(p, HyperParam)
            for attr_chain in p.usages:
                attr_chain.write_attrib(base=config, new_value=value)
        tf_session_opts = config.typed_dict.setdefault("tf_session_opts", {})
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
        gpu_opts = tf_session_opts.setdefault("gpu_options", tf_compat.v1.GPUOptions())
        if isinstance(gpu_opts, dict):
            gpu_opts = tf_compat.v1.GPUOptions(**gpu_opts)
        gpu_opts.visible_device_list = ",".join(map(str, sorted(gpu_ids)))
        return config

    def work(self):
        """
        Start the optimization.
        """
        print("Starting hyper param search. Using %i threads." % self.num_threads, file=log.v1)
        from returnn.tf.util.basic import get_available_gpu_devices
        from returnn.log import wrap_log_streams, StreamDummy
        from threading import Thread, Condition
        from returnn.util.basic import progress_bar, hms, is_tty

        class Outstanding:
            """
            Queue of outstanding work.
            """

            cond = Condition()
            threads = []  # type: typing.List[WorkerThread]
            population = []
            exit = False
            exception = None

        class WorkerThread(Thread):
            """
            Worker threader.
            """

            def __init__(self, gpu_ids):
                """
                :param set[int] gpu_ids:
                """
                super(WorkerThread, self).__init__(name="Hyper param tune train thread")
                self.gpu_ids = gpu_ids
                self.trainer = None  # type: typing.Optional[_IndividualTrainer]
                self.finished = False
                self.start()

            def cancel(self, join=False):
                """
                :param bool join:
                """
                with Outstanding.cond:
                    if self.trainer:
                        self.trainer.cancel_flag = True
                        if self.trainer.runner:
                            self.trainer.runner.cancel_flag = True
                if join:
                    self.join()

            def get_complete_frac(self):
                """
                :rtype: float
                """
                with Outstanding.cond:
                    if self.trainer and self.trainer.runner:
                        return self.trainer.runner.data_provider.get_complete_frac()
                return 0.0

            # noinspection PyMethodParameters
            def run(self_thread):
                """
                Run thread.
                """
                try:
                    while True:
                        with Outstanding.cond:
                            if Outstanding.exit or Outstanding.exception:
                                return
                            if not Outstanding.population:
                                self_thread.finished = True
                                Outstanding.cond.notify_all()
                                return
                            # noinspection PyShadowingNames
                            individual = Outstanding.population.pop(0)
                            self_thread.trainer = _IndividualTrainer(
                                optim=self, individual=individual, gpu_ids=self_thread.gpu_ids
                            )
                        self_thread.name = "Hyper param tune train thread on %r" % individual.name
                        self_thread.trainer.run()
                except Exception as exc:
                    with Outstanding.cond:
                        if not Outstanding.exception:
                            Outstanding.exception = exc or True
                        Outstanding.cond.notify_all()
                    for thread in Outstanding.threads:
                        if thread is not self_thread:
                            thread.cancel()
                    if not isinstance(exc, CancelTrainingException):
                        with Outstanding.cond:  # So that we don't mix up multiple on sys.stderr.
                            # This would normally dump it on sys.stderr so it's fine.
                            sys.excepthook(*sys.exc_info())

        best_individuals = []
        population = []
        num_gpus = len(get_available_gpu_devices())
        print("Num available GPUs:", num_gpus)
        num_gpus = num_gpus or 1  # Would be ignored anyway.
        interactive = is_tty()
        try:
            print(
                "Population of %i individuals (hyper param setting instances), running for %i evaluation iterations."
                % (self.num_individuals, self.num_iterations),
                file=log.v2,
            )
            for cur_iteration_idx in range(1, self.num_iterations + 1):
                print("Starting iteration %i." % cur_iteration_idx, file=log.v2)
                if cur_iteration_idx == 1:
                    population.append(Individual({p: p.get_default_value() for p in self.hyper_params}, name="default"))
                    population.append(
                        Individual({p: p.get_initial_value() for p in self.hyper_params}, name="canonical")
                    )
                population.extend(
                    self.get_population(
                        iteration_idx=cur_iteration_idx, num_individuals=self.num_individuals - len(population)
                    )
                )
                if cur_iteration_idx > 1:
                    self.cross_over(population=population, iteration_idx=cur_iteration_idx)
                if cur_iteration_idx == 1 and self.dry_run_first_individual:
                    # Train first directly for testing and to see log output.
                    # Later we will strip away all log output.
                    print("Very first try with log output:", file=log.v2)
                    _IndividualTrainer(optim=self, individual=population[0], gpu_ids={0}).run()
                print("Starting training with thread pool of %i threads." % self.num_threads)
                iteration_start_time = time.time()
                with wrap_log_streams(StreamDummy(), also_sys_stdout=True, tf_log_verbosity="WARN"):
                    Outstanding.exit = False
                    Outstanding.population = list(population)
                    Outstanding.threads = [WorkerThread(gpu_ids={i % num_gpus}) for i in range(self.num_threads)]
                    try:
                        while True:
                            with Outstanding.cond:
                                if all([thread.finished for thread in Outstanding.threads]) or Outstanding.exception:
                                    break
                                complete_frac = max(
                                    len(population) - len(Outstanding.population) - len(Outstanding.threads), 0
                                )
                                complete_frac += sum([thread.get_complete_frac() for thread in Outstanding.threads])
                                complete_frac /= float(len(population))
                                remaining_str = ""
                                if complete_frac > 0:
                                    start_elapsed = time.time() - iteration_start_time
                                    total_time_estimated = start_elapsed / complete_frac
                                    remaining_estimated = total_time_estimated - start_elapsed
                                    remaining_str = hms(remaining_estimated)
                                if interactive:
                                    progress_bar(complete_frac, prefix=remaining_str, file=sys.__stdout__)
                                else:
                                    print(
                                        "Progress: %.02f%%" % (complete_frac * 100),
                                        "remaining:",
                                        remaining_str or "unknown",
                                        file=sys.__stdout__,
                                    )
                                    sys.__stdout__.flush()
                                Outstanding.cond.wait(1 if interactive else 10)
                        for thread in Outstanding.threads:
                            thread.join()
                    finally:
                        Outstanding.exit = True
                        for thread in Outstanding.threads:
                            thread.cancel(join=True)
                Outstanding.threads = []
                print("Training iteration elapsed time:", hms(time.time() - iteration_start_time))
                if Outstanding.exception:
                    raise Outstanding.exception
                assert not Outstanding.population
                print("Training iteration finished.")
                population.sort(key=lambda p: p.cost)
                del population[-self.num_kill_individuals :]
                best_individuals.extend(population)
                best_individuals.sort(key=lambda p: p.cost)
                del best_individuals[self.num_best :]
                population = best_individuals[: self.num_kill_individuals // 4] + population
                print(
                    "Current best setting, individual %s" % best_individuals[0].name, "cost:", best_individuals[0].cost
                )
                for p in self.hyper_params:
                    print(" %s -> %s" % (p.description(), best_individuals[0].hyper_param_mapping[p]))
        except KeyboardInterrupt:
            print("KeyboardInterrupt, canceled search.")

        print("Best %i settings:" % len(best_individuals))
        for individual in best_individuals:
            print("Individual %s" % individual.name, "cost:", individual.cost)
            for p in self.hyper_params:
                print(" %s -> %s" % (p.description(), individual.hyper_param_mapping[p]))


class _IndividualTrainer:
    def __init__(self, optim, individual, gpu_ids):
        """
        :param Optimization optim:
        :param Individual individual:
        :param set[int] gpu_ids:
        """
        self.optim = optim
        self.individual = individual
        self.runner = None  # type: typing.Optional[Runner]
        self.gpu_ids = gpu_ids
        self.cancel_flag = False

    def run(self):
        """
        Run the trainer.
        """
        if self.individual.cost is not None:
            return self.individual.cost
        start_time = time.time()
        hyper_param_mapping = self.individual.hyper_param_mapping
        print("Training %r using hyper params:" % self.individual.name, file=log.v2)
        for p in self.optim.hyper_params:
            print(" %s -> %s" % (p.description(), hyper_param_mapping[p]), file=log.v2)
        config = self.optim.create_config_instance(hyper_param_mapping, gpu_ids=self.gpu_ids)
        # init_train_from_config expects to have the train task
        config.set("task", "train")
        engine = Engine(config=config)
        train_data = StaticDataset.copy_from_dataset(self.optim.train_data)
        engine.init_train_from_config(config=config, train_data=train_data)
        # Not directly calling train() as we want to have full control.
        engine.epoch = 1
        train_data.init_seq_order(epoch=engine.epoch)
        batches = train_data.generate_batches(
            recurrent_net=engine.network.recurrent,
            batch_size=engine.batch_size,
            max_seqs=engine.max_seqs,
            max_seq_length=int(engine.max_seq_length),
            seq_drop=engine.seq_drop,
            shuffle_batches=engine.shuffle_batches,
            used_data_keys=engine.network.used_data_keys,
        )
        engine.updater.set_learning_rate(engine.learning_rate, session=engine.tf_session)
        trainer = Runner(engine=engine, dataset=train_data, batches=batches, train=True)
        self.runner = trainer
        if self.cancel_flag:
            raise CancelTrainingException("Trainer cancel flag is set")
        trainer.run(report_prefix="hyper param tune train %r" % self.individual.name)
        if not trainer.finalized:
            print("Trainer exception:", trainer.run_exception, file=log.v1)
            raise trainer.run_exception
        cost = trainer.score["cost:output"]
        print(
            "Individual %s:" % self.individual.name,
            "Train cost:",
            cost,
            "elapsed time:",
            hms_fraction(time.time() - start_time),
            file=self.optim.log,
        )
        self.individual.cost = cost


class _AttribOrKey:
    ColTypeConfig = Config
    ColTypeDict = dict
    ColTypeObj = object

    def __init__(self, key, col_type):
        """
        :param str|object key:
        :param type[object]|type[dict] col_type:
        """
        self.key = key
        self.col_type = col_type

    def __str__(self):
        if self.col_type is self.ColTypeConfig:
            return "%s" % self.key
        if self.col_type is self.ColTypeDict:
            return "[%r]" % self.key
        if self.col_type is self.ColTypeObj:
            return ".%s" % self.key
        raise Exception("invalid col_type %r" % self.col_type)

    def get(self, parent):
        """
        :param object|dict|Config parent:
        :rtype: dict|object|HyperParam
        """
        if self.col_type is self.ColTypeConfig:
            return parent.typed_dict[self.key]
        if self.col_type is self.ColTypeDict:
            return parent[self.key]
        if self.col_type is self.ColTypeObj:
            return getattr(parent, self.key)
        raise Exception("invalid col_type %r" % self.col_type)

    def set(self, parent, new_value):
        """
        :param object|dict|Config parent:
        :param new_value:
        """
        if self.col_type is self.ColTypeConfig:
            parent.typed_dict[self.key] = new_value
            return
        if self.col_type is self.ColTypeDict:
            parent[self.key] = new_value
            return
        if self.col_type is self.ColTypeObj:
            setattr(parent, self.key, new_value)
            return
        raise Exception("invalid col_type %r" % self.col_type)


class _AttrChain:
    def __init__(self, base):
        """
        :param object|dict base:
        """
        self.base = base
        self.chain = []  # type: typing.List[_AttribOrKey]
        self.value = base  # type: typing.Union[HyperParam,object]

    def __str__(self):
        return "".join(map(str, self.chain))

    def __repr__(self):
        return "<%s %r %r>" % (self.__class__.__name__, self.chain, self.value)

    def get_extended_chain(self, attr):
        """
        :param _AttribOrKey attr:
        :rtype: _AttrChain
        """
        sub_chain = _AttrChain(base=self.base)
        sub_chain.chain = list(self.chain)
        sub_chain.chain.append(attr)
        sub_chain.value = attr.get(self.value)
        return sub_chain

    def write_attrib(self, base, new_value):
        """
        :param object|dict|Config base:
        :param new_value:
        """
        obj = base
        assert len(self.chain) >= 1
        for attr in self.chain[:-1]:
            obj = attr.get(obj)
        self.chain[-1].set(obj, new_value)


def hash_str_djb2(s):
    """
    :param str s:
    :rtype: int
    """
    v = 5381
    for x in s:
        v = ((v << 5) + v) + ord(x)
        v = v & 0xFFFFFFFF
    return v


def hash_seq(ls):
    """
    :param list|tuple ls:
    :rtype: int
    """
    v = 5381
    for x in ls:
        v = 1000003 * v + hash_obj(x)
        v = v & 0xFFFFFFFF
    return v


def hash_int(x):
    """
    :param int x:
    :rtype: int
    """
    return ((x << 11) + x) & 0xFFFFFFFF


def hash_obj(x):
    """
    :param tuple|list|str|_AttribOrKey|_AttrChain x:
    :rtype: int
    """
    if isinstance(x, (list, tuple)):
        return hash_seq(x)
    if isinstance(x, str):
        return hash_str_djb2(x)
    if isinstance(x, _AttribOrKey):
        return hash_str_djb2(x.key)
    if isinstance(x, _AttrChain):
        return hash_seq(x.chain)
    if isinstance(x, int):
        return hash_int(x)
    raise TypeError("invalid type %s" % type(x))
