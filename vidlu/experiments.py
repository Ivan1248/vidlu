import os
import re
import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import typing as T

import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

# from torch.utils.tensorboard import SummaryWriter

from vidlu import factories
import vidlu.modules as vm
from vidlu.training import Trainer, CheckpointManager, EpochLoop, IterState
from vidlu.utils.misc import indent_print
import vidlu.utils.distributed as vud
from vidlu.utils.logger import Logger
from vidlu.utils.path import to_valid_path
from vidlu.utils.misc import try_input, Stopwatch, query_user


DEFAULT_INTERACT_SHORTCUTS: T.Mapping[str, str] = {"i": "embed()", "skip": "loop.terminate()"}

# TODO: logger instead of verbosity

@dataclass
class TrainingExperimentFactoryArgs:
    data: str
    input_adapter: str
    model: str
    trainer: str
    metrics: str
    params: T.Optional[str]
    attach: str
    imports: str
    pre: str
    experiment_suffix: str
    resume: T.Optional[T.Literal["strict", "?", "best", "restart"]]
    device: T.Optional[torch.device]
    verbosity: int
    deterministic: bool
    factory_version: int
    distributed: T.Optional[bool]


# Component factories (or factory wrappers) ########################################################


def get_report_iters(eval_count, iter_count, type_=set):
    """Evenly distributes 0-based iteration indices."""
    if eval_count > iter_count:
        return type_(range(eval_count))
    return type_(np.unique(np.linspace(0.5, iter_count + 0.5, eval_count + 1, dtype=int)[1:] - 1))


def to_dhm_str(time):
    time = int(time)
    d, time_d = divmod(time, 24 * 60 * 60)
    h, time_h = divmod(time_d, 60 * 60)
    m = time_h // 60
    return " ".join([f"{v}{k}" for k, v in dict(d=d, h=h, m=m).items() if v > 0])


# def find_best_epoch(epoch_to_main_metrics, metrics_to_perf):
#     epochs = list(epoch_to_main_metrics.keys())
#     perfs = list(map(metrics_to_perf, epoch_to_main_metrics.values()))
#     argmax = np.argmax(perfs)
#     return epochs[argmax]


def report_metrics(state: IterState, is_training: bool, metrics: dict, epoch: int, epoch_count: int,
                   split_name=None, line_width=120,
                   special_format: T.Mapping[str, T.Callable[[str, T.Any], str]] = None,
                   prefix=None, array_prec=2, scalar_prec=4, logger: Logger = None):
    def fmt(v, scalar_prec=scalar_prec):
        with np.printoptions(precision=array_prec, threshold=4 if is_training else None,
                             linewidth=line_width, floatmode='maxprec_equal', suppress=True):
            if isinstance(v, (float, np.floating)):
                return (f"{v:.{scalar_prec}f}".lstrip('0') if v >= 1e-3 else
                        f"{v:.2e}")
            elif isinstance(v, dict):
                return "{" + ", ".join(f"{k}: {fmt(val, scalar_prec=array_prec)}" for k, val in v.items()) + "}"
            elif isinstance(v, np.ndarray) and v.ndim > 1:
                return f"\n{v}"
            else:
                return re.sub(r'(^|[\s\[\(])0\.', r'\1.', str(v))

    default_format = lambda k, v: f'{k}={fmt(v)}'
    special_format = special_format or {}

    def make_eval_str(metrics):
        parts, line_len = [], 0
        for k, v in metrics.items():
            parts.append(special_format.get(k, default_format)(k, v))
            if len(lines := parts[-1].splitlines()) > 1 or \
                    line_len + len(parts[-1]) > line_width:
                line_len = len(lines[-1])
                parts[-1] = f"\n{parts[-1]}"
        return ', '.join(parts)

    with indent_print():
        epoch_fmt, iter_fmt = f'{len(str(epoch_count))}d', f'{len(str(state.batch_count))}d'
        iter_ = state.iteration % state.batch_count
        if prefix is None:
            prefix = (f'{format(epoch + 1, epoch_fmt)}.{format(iter_ % state.batch_count + 1, iter_fmt)}' if is_training else 
                      f'{format(epoch + 1, epoch_fmt)} {split_name or "(?)"}')
        logger.log(f"{prefix}: {make_eval_str(metrics)}")
        # logger.log(f"Epoch to performance: {cpman.id_to_perf}")


class TrainingCallback:
    """Base for callbacks with auto-wiring by naming convention.

    Subclasses define methods like `on_training_epoch_started`, `on_evaluation_iter_completed`.
    These are auto-bound to `trainer.training.epoch_started`, `trainer.evaluation.iter_completed`, etc.
    """

    def attach(self, trainer: Trainer):
        """Bind handlers by naming convention: on_<loop>_<event> -> trainer.<loop>.<event>."""
        self.detach()
        self.trainer = trainer
        self._handles = []
        for name in dir(self):
            if not name.startswith("on_"):
                continue
            parts = name.split("_", 2)  # ["on", loop, event_name]
            if len(parts) < 3:
                continue
            loop = getattr(trainer, parts[1], None)
            event = getattr(loop, parts[2], None) if loop else None
            if event is not None:
                self._handles.append(event.add_handler(getattr(self, name)))

    def detach(self):
        """Remove handlers and drop trainer reference."""
        for h in getattr(self, "_handles", []):
            h.remove()
        self._handles = []
        self.trainer = None


class ProgressMonitor(TrainingCallback):
    def __init__(self, logger: Logger, eval_count, epoch_count, min_train_report_count=800,
                 line_width=120, special_format=None):
        self.logger = logger
        self.eval_count = eval_count
        self.epoch_count = epoch_count
        self.min_train_report_count = min_train_report_count
        self.line_width = line_width
        self.special_format = special_format or {}
        
        self.eval_epochs = get_report_iters(eval_count, epoch_count)
        self._report_iters = None  # Computed on first epoch when batch_count is known
        self.epoch_time = -1
        self.inter_epoch_time = -1
        self.eval_time = -1
        self.sw_epoch = Stopwatch()
        self.sw_inter_epoch = Stopwatch()
        self.sw_eval = Stopwatch()

    def on_training_epoch_started(self, state: IterState):
        """Restarts epoch time measurement, prints the estimated remaining time of training /
        evaluation and other information."""
        es = state
        self.sw_epoch.reset().start()
        
        if self._report_iters is None:
            report_count = max(1, self.min_train_report_count // self.epoch_count)
            self._report_iters = get_report_iters(report_count, es.batch_count)
        
        time_left_training = (1 - es.epoch / es.max_epochs) * (es.max_epochs * self.epoch_time)
        time_left = time_left_training + (1 - es.epoch / es.max_epochs) * (self.eval_count * self.eval_time)
        
        info_str = (f"Epoch {es.epoch + 1}/{es.max_epochs}:"
                    + f" {es.batch_count} batches,"
                    + f" lr=({', '.join(f'{x:.2e}' for x in self.trainer.lr_scheduler.get_last_lr())}),"
                    + f" devices={{{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}}}")
        if self.epoch_time > 0:
            info_str += f", left {to_dhm_str(time_left)} ({self.epoch_time:0.0f}s+{self.eval_time:0.0f}s per epoch)"
        self.logger.log(info_str)

    def on_training_epoch_completed(self, state: IterState):
        self.epoch_time = self.sw_epoch.time
        self.sw_inter_epoch.reset().start()

    def on_training_iter_completed(self, state: IterState):
        iter_ = state.iteration % state.batch_count
        if iter_ in self._report_iters:
            metrics = self.trainer.get_metric_values(reset=True)
            report_metrics(state, is_training=True, metrics=metrics, 
                           epoch=self.trainer.training.state.epoch, epoch_count=self.epoch_count,
                           line_width=self.line_width, special_format=self.special_format, 
                           logger=self.logger)

    def on_evaluation_epoch_started(self, state: IterState):
        self.inter_epoch_time = self.sw_inter_epoch.time
        self.sw_eval.reset().start()

    def on_evaluation_epoch_completed(self, state: IterState):
        self.eval_time = self.sw_eval.time
        split_name = getattr(state, 'split_name', None)
        metrics = self.trainer.get_metric_values(reset=True)
        report_metrics(state, is_training=False, metrics=metrics,
                       epoch=self.trainer.training.state.epoch, epoch_count=self.epoch_count,
                       split_name=split_name, line_width=self.line_width, 
                       special_format=self.special_format, logger=self.logger)


class ValidationCheckpointHandler(TrainingCallback):
    def __init__(self, data, cpman: CheckpointManager, main_metrics: T.Sequence[str], 
                 eval_count, epoch_count, logger: Logger, checkpoint_split_prefix: str | None = None):
        self.data = data
        self.cpman = cpman
        self.main_metrics = main_metrics
        self.eval_epochs = get_report_iters(eval_count, epoch_count)
        self.logger = logger
        self.checkpoint_split_prefix = checkpoint_split_prefix

    def on_training_epoch_completed(self, state: IterState):
        if state.epoch not in self.eval_epochs:
            return
            
        checkpoint_saved = False
        for name, ds in sorted(self.data.items()):
            if name.startswith("val"):
                # Run evaluation on the validation set
                es_val = self.trainer.eval(ds, split_name=name)
                
                should_checkpoint = (
                    not checkpoint_saved and 
                    (self.checkpoint_split_prefix is None or name.startswith(self.checkpoint_split_prefix))
                )
                if should_checkpoint:
                    main_metric_name = self.main_metrics[0] if len(self.main_metrics) > 0 else next(
                        iter(es_val.metrics.keys()))
                    self.cpman.save(self.trainer.state_dict(),
                               summary=dict(logger=self.logger.state_dict(),
                                            perf=es_val.metrics[main_metric_name],
                                            log="\n".join(self.logger.lines),
                                            epoch=state.epoch))
                    checkpoint_saved = True



class InteractiveController(TrainingCallback):
    def __init__(self, data, cpman, logger, main_metrics, interact_shortcuts=DEFAULT_INTERACT_SHORTCUTS):
        self.data = data
        self.cpman = cpman
        self.logger = logger
        self.main_metrics = main_metrics
        self.interact_shortcuts = dict() if interact_shortcuts is None else interact_shortcuts
        self.sleepiness = 0

        def populate_interactive_shell_namespace():
            import vidlu.utils.presentation.visualization as visualization
            from IPython import embed
            self.__dict__.update({"visualization": visualization, "embed": embed})
        populate_interactive_shell_namespace()

    def interact(self, state: IterState, loop: EpochLoop):
        if (optional_input := try_input()) is None:
            return

        namespace = {**vars(self).copy(), "state": state, "loop": loop}
        cmd = self.interact_shortcuts.get(optional_input, optional_input)
        print(f"Iteration: {state.iteration}, namespace: " + ", ".join(namespace.keys()))
        try:
            exec(cmd, globals(), namespace)
            for k in vars(self).keys():
                if k in namespace:
                    setattr(self, k, namespace[k])
        except Exception as e:
            print(f'Cannot execute "{optional_input}". Error:\n{e}.')

    def on_training_epoch_started(self, state: IterState):
        if self.sleepiness > 0:
            print(f"Warning: {self.sleepiness}s of sleep per epoch.")

    def on_training_iter_completed(self, state: IterState):
        self.interact(state, loop=self.trainer.training)
        if self.sleepiness > 0:
            time.sleep(self.sleepiness / state.batch_count)

    def on_evaluation_iter_completed(self, state: IterState):
        self.interact(state, loop=self.trainer.evaluation)
        if self.sleepiness > 0:
            time.sleep(self.sleepiness / state.batch_count)


def define_training_loop_actions(
        trainer: Trainer,
        cpman: CheckpointManager, 
        data: dict[str, T.Sequence], 
        logger: Logger, 
        main_metrics: T.Sequence[str],
        eval_count: int | None = None,
        min_train_report_count: int = 800, 
        interact_shortcuts: dict[str, str] | None = DEFAULT_INTERACT_SHORTCUTS,
        special_format: dict[str, callable] = {'mem': lambda k, v: f'{v}MiB', 'freq': lambda k, v: f'{v:.1f}/s',
                                               'freq_max': lambda k, v: f'freq_max={k, v:.1f}'},
        line_width: int = 120, 
        checkpoint_split_prefix: str | None = None) -> T.Sequence[TrainingCallback]:
    """
    Args:
        checkpoint_split_prefix: Prefix for split to use for checkpointing (default: None).
            If None, uses first evaluated split (current behavior).
            If provided (e.g., 'test'), checkpoints are saved only when evaluating splits
            starting with that prefix. This ensures checkpoints use metrics from the intended split.
    """
    if trainer.eval_count is not None:
        if eval_count is not None and eval_count != trainer.eval_count:
            raise ValueError("eval_count and trainer.eval_count cannot both be set.")
        eval_count = trainer.eval_count
        if 'VIDLU_EVAL_COUNT' in os.environ:
            warnings.warn(f"Overriding VIDLU_EVAL_COUNT={os.environ['VIDLU_EVAL_COUNT']} with"
                          f"trainer.eval_count={trainer.eval_count} to ensure proper training behavior.")
    elif eval_count is None:
        eval_count = int(os.environ.get('VIDLU_EVAL_COUNT', 200))

    progress_mon = ProgressMonitor(
        logger=logger, eval_count=eval_count, epoch_count=trainer.epoch_count, 
        min_train_report_count=min_train_report_count, line_width=line_width, 
        special_format=special_format)
    validator = ValidationCheckpointHandler(
        data=data, cpman=cpman, main_metrics=main_metrics, eval_count=eval_count, 
        epoch_count=trainer.epoch_count, logger=logger, 
        checkpoint_split_prefix=checkpoint_split_prefix)
    interactor = InteractiveController(data=data, cpman=cpman, logger=logger, 
                                       main_metrics=main_metrics, interact_shortcuts=interact_shortcuts)

    # Attach components
    progress_mon.attach(trainer)
    validator.attach(trainer)
    interactor.attach(trainer)

    return [progress_mon, validator, interactor]

# Experiment #######################################################################################

def get_device(device_id: T.Optional[T.Union[str, int]], distributed: bool):
    if distributed:
        if device_id is None:
            rank = vud.get_local_rank()
            return torch.device(rank)
        else:
            warnings.warn("The device argument should be set to None for distributed"
                          + " training on multiple devices.")
    if device_id is None:
        if torch.cuda.device_count() == 0 \
                and not query_user("No GPU found. Do you want to use a CPU?",
                                   default='y', timeout=10):
            raise RuntimeError("No GPU available.")
        return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
    elif device_id == "auto":
        from vidlu import gpu_utils
        return torch.device(
            gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))
    else:
        return torch.device(device_id)


def get_experiment_name(training_args):
    a = training_args
    learner_name = to_valid_path(f"{a.input_adapter}/{a.model}/{a.trainer}"
                                 + (f"/{a.params}" if a.params else ""), split_long_names=True)
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{to_valid_path(a.data, split_long_names=True)}/{learner_name}/{expsuff}'
    return experiment_id


def create_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_root):
    a = training_args
    experiment_id = get_experiment_name(training_args)
    cpman = CheckpointManager(
        checkpoints_root, experiment_name=experiment_id, experiment_info=training_args,
        separately_saved_state_parts=("model",), n_best_kept=1,
        start_mode=('start' if a.resume is None else
                    'restart' if a.resume == "restart" else
                    'resume_or_start' if a.resume == "?" else
                    'resume'),
        perf_func=lambda s: s.get('perf', 0),
        log_func=lambda s: s.get('log', ""),
        name_suffix_func=lambda s: f"{s['epoch']}_{s['perf']:.3f}")
    return cpman


def load_parameters(model, params_str, params_dir):
    parameters, dest = factories.get_translated_parameters(params_str=params_str,
                                                           params_dir=params_dir)
    module = vm.get_submodule(model, dest)
    try:
        module.load_state_dict(parameters, strict=True)
    except RuntimeError as e:
        warnings.warn(str(e))
        module.load_state_dict(parameters, strict=False)


def _check_dirs(dirs):
    for name in ['datasets', 'cache', 'saved_states', 'pretrained']:
        dirs_ = getattr(dirs, name)
        if isinstance(dirs_, (str, Path)):
            dirs_ = [dirs_]
        for d in dirs_:
            if d is None or not Path(d).is_dir():
                raise NotADirectoryError(f"dirs.{name}={getattr(dirs, name)} is not"
                                         f" a directory path or a sequence thereof.")


def get_device_and_distributed_flag(device_id, distributed_arg: bool | None):
    print(f"{distributed_arg=}")
    if distributed_arg is None:
        distributed_arg = vud.distributed_is_enabled()
    print(f"enabled {distributed_arg=}")
    if distributed_arg:
        print(f"\nDistributed training: global rank: {vud.get_global_rank()},"
              + f" local rank: {vud.get_local_rank()},"
              + f" number of processes: {vud.get_global_size()}")
    device = get_device(device_id, distributed_arg)
    print(f"device: {device}")
    return device, distributed_arg


def make_model_distributed(model):
    model = DistributedDataParallel(model, device_ids=[vud.get_local_rank()])
    for m in model.modules():
        if type(m).__name__.startswith("Batch"):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            break
    return model


def init_model(model_str: str, input_adapter_str: str, verbosity, device, distributed: bool,
               prep_dataset, namespace: dict):
    print(model_str)
    with Stopwatch() as sw:
        model = factories.get_model(model_str, input_adapter_str=input_adapter_str,
                                    prep_dataset=prep_dataset, device=device, namespace=namespace,
                                    verbosity=verbosity)
        if distributed:
            model = make_model_distributed(model)
    print(f"Model initialized in {sw.time:.2f} s.")
    return model


def get_trainer_and_metrics(trainer_str, metrics_str, deterministic, distributed, first_ds, model,
                            verbosity, namespace: dict, *, data=None):
    print(trainer_str)
    trainer = factories.get_trainer(trainer_str, model=model, data=data, verbosity=verbosity,
                                    deterministic=deterministic, distributed=distributed,
                                    namespace=namespace)
    # TODO: distributed metrics
    metrics, main_metrics = factories.get_metrics(metrics_str, trainer, dataset=first_ds)
    for m in metrics:
        trainer.metrics.append(m)
    return trainer, (metrics, main_metrics)


@dataclass
class TrainingExperiment:
    model: nn.Module
    trainer: Trainer
    data: Namespace
    cpman: CheckpointManager
    attachments: T.Sequence[object]
    logger: Logger

    @staticmethod
    def from_args(training_args: TrainingExperimentFactoryArgs, dirs):
        _check_dirs(dirs)
        logger = Logger()
        a = training_args

        experiment = Namespace()
        factory_namespace = factories.make_namespace(training_args.imports, training_args.pre)
        factory_namespace.update(dirs=dirs, experiment=experiment)

        with indent_print("\nSetting device and setting up distributed training..."):
            device, distributed = get_device_and_distributed_flag(a.device, a.distributed)

        with indent_print('\nInitializing checkpoint manager...'):
            experiment.cpman = cpman = create_checkpoint_manager(a, dirs.saved_states)

        try:
            with indent_print('\nInitializing data...'):
                print(a.data)
                with Stopwatch() as sw:
                    experiment.data = data = factories.get_prepared_data_for_trainer(
                        a.data, dirs.datasets, dirs.cache, namespace=factory_namespace,
                        factory_version=a.factory_version)
                print(f"Data initialized in {sw.time:.2f} s.")
            first_ds = next(iter(data.values()))

            with indent_print('\nInitializing model...'):
                experiment.model = model = init_model(
                    a.model, a.input_adapter, a.verbosity, device, distributed, first_ds,
                    namespace=factory_namespace)

            with indent_print('\nInitializing trainer and evaluation...'):
                trainer, (metrics, main_metrics) = get_trainer_and_metrics(
                    a.trainer, a.metrics, a.deterministic, distributed, first_ds, model,
                    a.verbosity, namespace=factory_namespace, data=experiment.data)
                # Trainer already receives data via factory; avoid post-construction mutation.
                experiment.trainer = trainer

            define_training_loop_actions(trainer, cpman, experiment.data, logger,
                                         main_metrics=main_metrics)
        except Exception:
            raise
        finally:
            resuming_required = cpman.resuming_required
            if resuming_required:
                state, summary, _ = (cpman.load_best if a.resume == "best" else cpman.load_last)(
                    map_location=device)
                # TODO: remove backward compatibility
                logger.load_state_dict(summary.get('logger', summary))
                logger.print_all()

        experiment.attachments = eval(training_args.attach, dict(), factory_namespace)

        if resuming_required:
            trainer.load_state_dict(state)
        elif a.params is not None:
            with indent_print("\nLoading parameters..."):
                print(a.params)
                load_parameters(model, a.params, dirs.pretrained)

        return TrainingExperiment(**experiment.__dict__, logger=logger)


class TrainingExperimentBuilder(Namespace):
    def build(self) -> TrainingExperiment:
        required = ["model", "trainer", "data", "cpman", "attachments", "logger", "callbacks"]
        missing = [key for key in required if not hasattr(self, key)]
        if missing:
            raise ValueError(f"Missing required attributes: {', '.join(missing)}")
        return TrainingExperiment(model=self.model, trainer=self.trainer, data=self.data, 
                                  cpman=self.cpman, attachments=self.attachments, logger=self.logger, 
                                  callbacks=self.callbacks)
