# -*- coding: utf-8 -*-
import gc
import importlib
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from inspect import signature
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainerConfig, GPTArgs, XttsAudioConfig
from coqpit import Coqpit
import json
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from trainer.callbacks import TrainerCallback
from trainer.generic_utils import (
    KeepAverage,
    count_parameters,
    get_experiment_folder_path,
    get_git_branch,
    isimplemented,
    remove_experiment_folder,
    set_partial_state_dict,
    to_cuda,
)
from trainer.io import (
    copy_model_files,
    get_last_checkpoint,
    load_fsspec,
    save_best_model,
    save_checkpoint,
)
from trainer.logging import ConsoleLogger, DummyLogger, logger_factory
from trainer.trainer_utils import (
    get_optimizer,
    get_scheduler,
    is_apex_available,
    print_training_env,
    setup_torch_training_env,
)
from trainer.utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from trainer.utils.distributed import (
    get_rank,
    init_distributed,
    rank_zero_logger_info,
    rank_zero_only,
)

from trainer_alltalk.WarmUpScheduler import WarmUpScheduler

logger = logging.getLogger("trainer")

if is_apex_available():
    from apex import amp  # pylint: disable=import-error


@dataclass
class TrainerConfig(Coqpit):
    """Config fields tweaking the Trainer for a model.
    A ````ModelConfig```, by inheriting ```TrainerConfig``` must be defined for using 👟.
    Inherit this by a new model config and override the fields as needed.
    All the fields can be overridden from comman-line as ```--coqpit.arg_name=value```.

    Example::

        Run the training code by overriding the ```lr``` and ```plot_step``` fields.

        >>> python train.py --coqpit.plot_step=22 --coqpit.lr=0.001

        Defining a model using ```TrainerConfig```.

        >>> from trainer import TrainerConfig
        >>> class MyModelConfig(TrainerConfig):
        ...     optimizer: str = "Adam"
        ...     lr: float = 0.001
        ...     epochs: int = 1
        ...     ...
        >>> class MyModel(nn.module):
        ...    def __init__(self, config):
        ...        ...
        >>> model = MyModel(MyModelConfig())

    """

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    save_on_interrupt: bool = field(
        default=True, metadata={"help": "Save checkpoint on interrupt (Ctrl+C). Defaults to True"}
    )
    log_model_step: int = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(
        default=0, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 0"}
    )
    target_loss: str = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision to use in mixed precision training. `fp16` for float16 and `bf16` for bfloat16. Defaults to 'f16'"
        },
    )
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float = field(
        default=0.0, metadata={"help": "Gradient clipping value. Disabled if <= 0. Defaults to 0.0"}
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: Union[float, List[float]] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: Union[str, List[str]] = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: Union[Dict, List[Dict]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: Union[str, List[str]] = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: Dict = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False."
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )


@dataclass
class TrainerArgs(Coqpit):
    """Trainer arguments that can be accessed from the command line.

    Examples::
        >>> python train.py --restore_path /path/to/checkpoint.pth
    """

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting the best loss. If not specified, the latest best model in continue path is used"
        },
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "Use DDP in distributed training. It is to set in `distribute.py`. Do not set manually."},
    )
    use_accelerate: bool = field(default=False, metadata={"help": "Use HF Accelerate as the back end for training."})
    grad_accum_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. It is used to accumulate gradients over multiple batches."
        },
    )
    overfit_batch: bool = field(default=False, metadata={"help": "Overfit a single batch for debugging."})
    skip_train_epoch: bool = field(
        default=False,
        metadata={"help": "Skip training and only run evaluation and test."},
    )
    start_with_eval: bool = field(
        default=False,
        metadata={"help": "Start with evaluation and test."},
    )
    small_run: int = field(
        default=None,
        metadata={
            "help": "Only use a subset of the samples for debugging. Set the number of samples to use. Defaults to None. "
        },
    )
    gpu: int = field(
        default=None, metadata={"help": "GPU ID to use if ```CUDA_VISIBLE_DEVICES``` is not set. Defaults to None."}
    )
    # only for DDP
    rank: int = field(default=0, metadata={"help": "Process rank in a distributed training. Don't set manually."})
    group_id: str = field(
        default="", metadata={"help": "Process group id in a distributed training. Don't set manually."}
    )


class Trainer:
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        args: TrainerArgs,
        config: Coqpit,
        output_path: str,
        c_logger: ConsoleLogger = None,
        dashboard_logger: "Logger" = None,
        model: nn.Module = None,
        get_model: Callable = None,
        get_data_samples: Callable = None,
        train_samples: List = None,
        eval_samples: List = None,
        test_samples: List = None,
        train_loader: DataLoader = None,
        eval_loader: DataLoader = None,
        training_assets: Dict = {},
        parse_command_line_args: bool = True,
        callbacks: Dict[str, Callable] = {},
        gpu: int = None,
        warmup: bool = False,
    ) -> None:
        """Simple yet powerful 🐸💬 TTS trainer for PyTorch. It can train all the available `tts` and `vocoder` models
        or easily be customized.

        Notes:

            Supports Automatic Mixed Precision training. If `Apex` is availabe, it automatically picks that, else
            it uses PyTorch's native `amp` module. `Apex` may provide more stable training in some cases.

        Args:

            args (Union[Coqpit, Namespace]): Training arguments parsed either from console by `argparse` or `TrainerArgs`
                config object.

            config (Coqpit): Model config object. It includes all the values necessary for initializing, training, evaluating
                and testing the model.

            output_path (str): Path to the output training folder. All the files are saved under thi path.

            c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
                console logger is used. Defaults to None.

            dashboard_logger Union[TensorboardLogger, WandbLogger]: Dashboard logger. If not provided, the tensorboard logger is used.
                Defaults to None.

            model (nn.Module, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
                initializes a model from the provided config. Defaults to None.

            get_model (Callable):
                A function that returns a model. It is used to initialize the model when `model` is not provided.
                It either takes the config as the only argument or does not take any argument.
                Defaults to None

            get_data_samples (Callable):
                A function that returns a list of training and evaluation samples. Used if `train_samples` and
                `eval_samples` are None. Defaults to None.

            train_samples (List):
                A list of training samples used by the model's `get_train_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            eval_samples (List):
                A list of evaluation samples used by the model's `get_eval_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            train_loader (DataLoader):
                A pytorch data loader object for training epochs. Leave as None if you want it to be made during training. Defaults to None.

            eval_loader (DataLoader):
                A pytorch data loader object for evaluation epochs. Leave as None to be generated during training. Defaults to None.

            test_samples (List):
                A list of test samples used by the model's `get_test_data_loader` to init the `dataset` and the
                `data_loader`. If None, the ```model.test_run()``` is expected to load the data. Defaults to None.

            training_assets (Dict):
                A dictionary of assets to be used at training and passed to the model's ```train_log(), eval_log(), get_data_loader()```
                during training. It can include  `AudioProcessor` or/and `Tokenizer`. Defaults to {}.

            parse_command_line_args (bool):
                If true, parse command-line arguments and update `TrainerArgs` and model `config` values. Set it
                to false if you parse the arguments yourself. Defaults to True.

            callbacks (Dict[str, Callable]):
                A dictionary of callbacks to be used during training. The keys are the callback names and the values

            gpu (int):
                GPU ID to use for training If "CUDA_VISIBLE_DEVICES" is not set. Defaults to None.

        Example::

            Running trainer with a model.

            >>> args = TrainerArgs(...)
            >>> config = ModelConfig(...)
            >>> model = Model(config)
            >>> trainer = Trainer(args, config, output_path, model=model)
            >>> trainer.fit()

            TODO:
                - Wrap model for not calling .module in DDP.
                - Deepspeed integration
                - Profiler integration.
                - Overfitting to a batch.
                - TPU training
        """
        if parse_command_line_args:
            # parse command-line arguments to override TrainerArgs()
            args, coqpit_overrides = self.parse_argv(args)

            # get ready for training and parse command-line arguments to override the model config
            config, new_fields = self.init_training(args, coqpit_overrides, config)
        elif args.continue_path or args.restore_path:
            config, new_fields = self.init_training(args, {}, config)
        else:
            new_fields = {}

        # set the output path
        if args.continue_path:
            # use the same path as the continuing run
            output_path = args.continue_path
        else:
            # override the output path if it is provided
            output_path = config.output_path if output_path is None else output_path
            # create a new output folder name
            output_path = get_experiment_folder_path(config.output_path, config.run_name)
            os.makedirs(output_path, exist_ok=True)

        # copy training assets to the output folder
        copy_model_files(config, output_path, new_fields)

        # init class members
        self.args = args
        self.config = config
        self.output_path = output_path
        self.training_assets = training_assets
        self.grad_accum_steps = args.grad_accum_steps
        self.overfit_batch = args.overfit_batch
        self.skip_train_epoch = args.skip_train_epoch
        self.start_with_eval = args.start_with_eval
        self.warmup = warmup

        assert self.grad_accum_steps > 0, " [!] grad_accum_steps must be greater than 0."

        # setup logging
        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        # setup training environment
        self.use_cuda, self.num_gpus = self.setup_training_environment(args=args, config=config, gpu=gpu)

        # init loggers
        self.dashboard_logger, self.c_logger = self.init_loggers(self.config, output_path, dashboard_logger, c_logger)
        # self.c_logger.logger = logger

        if not self.config.log_model_step:
            self.config.log_model_step = self.config.save_step

        # make sure that start_with_eval is disabled if eval is disabled
        if not self.config.run_eval and self.start_with_eval:
            self.start_with_eval = False

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.restore_epoch = 0
        self.best_loss = {"train_loss": float("inf"), "eval_loss": float("inf") if self.config.run_eval else None}
        self.train_loader = None
        self.test_loader = None
        self.eval_loader = None

        self.keep_avg_train = None
        self.keep_avg_eval = None

        self.use_amp_scaler = (
            self.use_cuda
            if self.config.mixed_precision and self.config.precision == "fp16"
            else self.config.use_grad_scaler
        )

        if train_samples is not None:
            # use the provided samples
            self.train_samples = train_samples
            self.eval_samples = eval_samples
            self.test_samples = test_samples
        elif get_data_samples is not None:
            # run `get_data_samples` to init the data samples
            (  # pylint: disable=unbalanced-tuple-unpacking
                self.train_samples,
                self.eval_samples,
                self.test_samples,
            ) = self.run_get_data_samples(config, get_data_samples)
        else:
            # expecting to load the samples in `model.get_data_loader()`
            self.train_samples = None
            self.eval_samples = None
            self.test_samples = None

        # define custom train and eval loader
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # only use a subset of the samples if small_run is set
        self.setup_small_run(args.small_run)

        # init the model
        if model is None and get_model is None:
            raise ValueError("[!] `model` and `get_model` cannot both be None.")
        if model is not None:
            self.model = model
        else:
            self.run_get_model(self.config, get_model)

        # init model's training assets
        if isimplemented(self.model, "init_for_training"):
            self.model.init_for_training()

        # setup criterion
        self.criterion = self.get_criterion(self.model)

        # DISTRUBUTED
        if self.use_pt_ddp:
            rank_zero_logger_info(" > Using PyTorch DDP", logger)
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed_backend,
                self.config.distributed_url,
            )

        if self.use_cuda:
            self.model.cuda()
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    if isinstance(criterion, torch.nn.Module):
                        criterion.cuda()
            else:
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.cuda()

        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        # If multiple-optimizer setup with grad accumulation and without custom optimize method raise an error
        if (
            self.grad_accum_steps != 1
            and isinstance(self.optimizer, list)
            and not isimplemented(self.model, "optimize")
        ):
            raise ValueError(
                " [!] Coqui Trainer does not support grad_accum_steps for multiple-optimizer setup, please set grad_accum_steps to 1 or implement in your model a custom method called ´optimize` that need to deal with dangling gradients in multiple-optimizer setup!"
            )

        # CALLBACK
        self.callbacks = TrainerCallback()
        self.callbacks.parse_callbacks_dict(callbacks)
        self.callbacks.on_init_start(self)

        # init AMP
        if self.use_amp_scaler:
            if self.use_apex:
                self.scaler = None
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # restore model
        if self.args.restore_path:
            (self.model, self.optimizer, self.scaler, self.restore_step, self.restore_epoch) = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )
            self.scaler = torch.cuda.amp.GradScaler()

        # setup scheduler
        self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)
        self.scheduler = self.restore_scheduler(
            self.scheduler, self.args, self.config, self.restore_epoch, self.restore_step
        )

        if self.warmup:
            warmup_scheduler = WarmUpScheduler(
                optimizer=self.optimizer,
                total_epochs=self.config.epochs,
                warmup_lr=1e-7,
                target_lr=self.config.lr,
                after_scheduler=self.scheduler
            )
            self.scheduler = warmup_scheduler

        # DISTRIBUTED
        if self.use_pt_ddp:
            self.model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)

        # setup accelerator
        self.setup_accelerate()

        # count model size
        num_params = count_parameters(self.model)
        rank_zero_logger_info(f"\n > Model has {num_params} parameters", logger)

        self.callbacks.on_init_end(self)
        self.dashboard_logger.add_config(config)
        self.save_training_script()

    @property
    def use_apex(self):
        """Return True if using APEX."""
        return not self.args.use_accelerate and self._is_apex_available()

    @property
    def use_pt_ddp(self):
        """Return True if using PyTorch DDP."""
        return self.num_gpus > 1 and not self.use_accelerate

    @property
    def use_accelerate(self):
        """Return True if using HF Accelerate."""
        return self.args.use_accelerate

    def setup_accelerate(self):
        if self.use_accelerate:
            self.model, self.optimizer, self.train_loader, self.scheduler, self.accelerator = self.init_accelerate(
                model=self.model,
                optimizer=self.optimizer,
                training_dataloader=self.train_loader,
                scheduler=self.scheduler,
                grad_accum_steps=self.grad_accum_steps,
                mixed_precision=self.config.mixed_precision,
                precision=self.config.precision,
            )

    def prepare_accelerate_loader(self, data_loader):
        """Prepare the accelerator for the training."""
        if self.use_accelerate:
            return self.accelerator.prepare_data_loader(data_loader)
        return data_loader

    @staticmethod
    def init_accelerate(model, optimizer, training_dataloader, scheduler, grad_accum_steps, mixed_precision, precision):
        """Setup HF Accelerate for the training."""

        # check if accelerate is installed
        try:
            from accelerate import Accelerator  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install accelerate to use this feature.") from e

        _precision = precision if precision is not None else "f16" if mixed_precision else None
        if _precision == "float16":
            _precision = "f16"
        elif _precision == "float8":
            _precision = "f8"
        elif _precision == "bfloat16":
            _precision = "bf16"
        accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps, mixed_precision=_precision)
        if isinstance(model, torch.nn.Module):
            model = accelerator.prepare_model(model)

        if isinstance(optimizer, dict):
            for key, optim in optimizer.items():
                optimizer[key] = accelerator.prepare_optimizer(optim)
        elif isinstance(optimizer, list):
            for i, optim in enumerate(optimizer):
                optimizer[i] = accelerator.prepare_optimizer(optim)
        elif optimizer is not None:
            optimizer = accelerator.prepare_optimizer(optimizer)

        if isinstance(training_dataloader, torch.utils.data.DataLoader):
            training_dataloader = accelerator.prepare_data_loader(training_dataloader)

        if isinstance(scheduler, dict):
            for key, sched in scheduler.items():
                scheduler[key] = accelerator.prepare_scheduler(sched)
        elif isinstance(scheduler, list):
            for i, sched in enumerate(scheduler):
                scheduler[i] = accelerator.prepare_scheduler(sched)
        elif scheduler is not None:
            scheduler = accelerator.prepare_scheduler(scheduler)

        return model, optimizer, training_dataloader, scheduler, accelerator

    def save_training_script(self):
        """Save the training script to tracking dashboard and output path."""
        file_path = sys.argv[0]
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            self.dashboard_logger.add_artifact(file_or_dir=file_path, name=file_name, artifact_type="file")
            with open(file_path, "r", encoding="utf8") as f:
                self.dashboard_logger.add_text("training-script", f"{f.read()}", 0)
            shutil.copyfile(file_path, os.path.join(self.output_path, file_name))

    @staticmethod
    def parse_argv(args: Union[Coqpit, List]):
        """Parse command line arguments to init or override `TrainerArgs()`."""
        if isinstance(args, Coqpit):
            parser = args.init_argparse(arg_prefix="")
        else:
            train_config = TrainerArgs()
            parser = train_config.init_argparse(arg_prefix="")
        training_args, coqpit_overrides = parser.parse_known_args()
        args.parse_args(training_args)
        return args, coqpit_overrides

    @staticmethod
    def init_loggers(config: "Coqpit", output_path: str, dashboard_logger=None, c_logger=None):
        """Init console and dashboard loggers.
        Use the given logger if passed externally else use config values to pick the right logger.
        Return a dashboard logger only for the rank 0 process in DDP
        Define a console logger for each process in DDP

        Args:
            config (Coqpit): Model config.
            output_path (str): Output path to save the training artifacts.
            dashboard_logger (DashboardLogger): Object passed to the trainer from outside.
            c_logger (ConsoleLogger): Object passed to the trained from outside.

        Returns:
            Initialized dashboard_logger and console_logger objects.
        """
        c_logger = ConsoleLogger() if c_logger is None else c_logger

        # only allow dashboard logging for the main process in DDP mode
        if get_rank() > 0:
            return DummyLogger(), c_logger
        if dashboard_logger is None:
            dashboard_logger = logger_factory(config, output_path)
        return dashboard_logger, c_logger

    def setup_small_run(self, small_run: int = None):
        """Use a subset of samples for training, evaluation and testing."""
        if small_run is not None:
            logger.info("[!] Small Run, only using %i samples.", small_run)
            self.train_samples = None if self.train_samples is None else self.train_samples[:small_run]
            self.eval_samples = None if self.eval_samples is None else self.eval_samples[:small_run]
            self.test_samples = None if self.test_samples is None else self.test_samples[:small_run]

    @staticmethod
    def init_training(args: TrainerArgs, coqpit_overrides: Dict, config: Coqpit = None):
        """Initialize training and update model configs from command line arguments.

        Args:
            args (argparse.Namespace or dict like): Parsed trainer arguments.
            config_overrides (argparse.Namespace or dict like): Parsed config overriding arguments.
            config (Coqpit): Model config. If none, it is generated from `args`. Defaults to None.

        Returns:
            config (Coqpit): Config paramaters.
        """
        # set arguments for continuing training
        if args.continue_path:
            print("continue_path training from previous run.")
            args.config_path = os.path.join(args.continue_path, "config.json")
            args.restore_path, best_model = get_last_checkpoint(args.continue_path)
            if not args.best_path:
                args.best_path = best_model
            # use the same config
            if config:
                # Fixes https://github.com/coqui-ai/TTS/issues/3131
                # Dirty, but they are closed
                with open(args.config_path, 'r') as file:
                    data = json.load(file)
                    audio_data = data["audio"]
                    model_data = data["model_args"]

                    model_args = {}
                    for key, value in config["model_args"].items():
                        model_args[key] = model_data.get(key, value)

                    audio_args = {}
                    for key, value in config["audio"].items():
                        audio_args[key] = audio_data.get(key, value)

                    config_args = {}
                    for key, value in config.items():
                        config_args[key] = data.get(key, value)

                    config = GPTTrainerConfig(**config_args)
                    config["model_args"] = GPTArgs(**model_args)
                    config["audio"] = XttsAudioConfig(**audio_args)
            else:
                coqpit = Coqpit()
                coqpit.load_json(args.config_path)

        # override config values from command-line args
        # TODO: Maybe it is better to do it outside
        if len(coqpit_overrides) > 0:
            config.parse_known_args(coqpit_overrides, relaxed_parser=True)

        # update the config.json fields and copy it to the output folder
        new_fields = {}
        if args.rank == 0:
            if args.restore_path:
                new_fields["restore_path"] = args.restore_path
            new_fields["github_branch"] = get_git_branch()
        return config, new_fields

    @staticmethod
    def setup_training_environment(args, config, gpu):
        if platform.system() != "Windows":
            # https://github.com/pytorch/pytorch/issues/973
            import resource  # pylint: disable=import-outside-toplevel

            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        # set and initialize Pytorch runtime
        use_cuda, num_gpus = setup_torch_training_env(
            args=args,
            cudnn_enable=config.cudnn_enable,
            cudnn_deterministic=config.cudnn_deterministic,
            cudnn_benchmark=config.cudnn_benchmark,
            use_ddp=args.use_ddp,
            training_seed=config.training_seed,
            allow_tf32=config.allow_tf32,
            gpu=gpu if args.gpu is None else args.gpu,
        )

        print_training_env(args, config)
        return use_cuda, num_gpus

    @staticmethod
    def run_get_model(config: Coqpit, get_model: Callable) -> nn.Module:
        """Run the `get_model` function and return the model.

        Args:
            config (Coqpit): Model config.

        Returns:
            nn.Module: initialized model.
        """
        if len(signature(get_model).sig.parameters) == 1:
            model = get_model(config)
        else:
            model = get_model()
        return model

    @staticmethod
    def run_get_data_samples(config: Coqpit, get_data_samples: Callable) -> nn.Module:
        if callable(get_data_samples):
            if len(signature(get_data_samples).sig.parameters) == 1:
                train_samples, eval_samples = get_data_samples(config)
            else:
                train_samples, eval_samples = get_data_samples()
            return train_samples, eval_samples
        return None, None

    def restore_model(
        self,
        config: Coqpit,
        restore_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]:
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (Coqpit): Model config.
            restore_path (str): Path to the restored training run.
            model (nn.Module): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.cuda.amp.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]: [description]
        """

        def _restore_list_objs(states, obj):
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            elif isinstance(obj, dict):
                for key, state in states.items():
                    obj[key].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        logger.info(" > Restoring from %s ...", os.path.basename(restore_path))
        checkpoint = load_fsspec(restore_path, map_location="cpu")

        try:
            logger.info(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            logger.info(" > Restoring Optimizer...")
            try:
                optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
            except (KeyError, TypeError, RuntimeError):
                logger.info(" > Optimizer is not compatible with the restored model.")
            if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
                logger.info(" > Restoring Scaler...")
                scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        except (KeyError, RuntimeError, ValueError):
            logger.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_partial_state_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        optimizer = self.restore_lr(config, self.args, model, optimizer)

        logger.info(" > Model restored from step %i", checkpoint["step"])
        restore_step = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
        restore_epoch = checkpoint["epoch"]
        torch.cuda.empty_cache()
        return model, optimizer, scaler, restore_step, restore_epoch

    def restore_lr(self, config, args, model, optimizer):
        # use the same lr if continue training
        if not args.continue_path:
            if isinstance(optimizer, list):
                for idx, optim in enumerate(optimizer):
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[idx]
            elif isinstance(optimizer, dict):
                for optim_name, optim in optimizer.items():
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[optim_name]
            else:
                for group in optimizer.param_groups:
                    group["lr"] = self.get_lr(model, config)
        return optimizer

    #########################
    # DATA LOADING FUNCTIONS
    #########################

    def _get_loader(
        self,
        model: nn.Module,
        config: Coqpit,
        assets: Dict,
        is_eval: str,
        samples: List,
        verbose: bool,
        num_gpus: int,
    ) -> DataLoader:
        if num_gpus > 1:
            if isimplemented(model.module, "get_data_loader"):
                loader = model.module.get_data_loader(
                    config,
                    assets,
                    is_eval,
                    samples,
                    verbose,
                    num_gpus,
                    self.args.rank,
                )
        else:
            if isimplemented(model, "get_data_loader"):
                loader = model.get_data_loader(
                    config=config, assets=assets, is_eval=is_eval, samples=samples, verbose=verbose, num_gpus=num_gpus
                )

        assert (
            len(loader) > 0
        ), " ❗ len(DataLoader) returns 0. Make sure your dataset is not empty or len(dataset) > 0. "
        return loader

    def get_train_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a training data loader.
        Call ```model.get_train_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=False```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_train_data_loader"):
                loader = self.model.module.get_train_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_train_data_loader"):
                loader = self.model.get_train_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            False,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_eval_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_eval_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_eval_data_loader"):
                loader = self.model.module.get_eval_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_eval_data_loader"):
                loader = self.model.get_eval_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_test_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_test_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_test_data_loader"):
                loader = self.model.module.get_test_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_test_data_loader"):
                loader = self.model.get_test_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def format_batch(self, batch: List) -> Dict:
        """Format the dataloader output and return a batch.

        1. Call ```model.format_batch```.
        2. Pass the batch to the Device.
        3. Call ```model.format_batch_on_device```.

        Args:
            batch (List): Batch returned by the dataloader.

        Returns:
            Dict: Formatted batch.
        """
        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch(batch)
            else:
                batch = self.model.format_batch(batch)
        except NotImplementedError:
            pass

        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]

        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch_on_device(batch)
            else:
                batch = self.model.format_batch_on_device(batch)
        except NotImplementedError:
            pass
        return batch

    ######################
    # TRAIN FUNCTIONS
    ######################

    @staticmethod
    def master_params(optimizer: torch.optim.Optimizer):
        """Generator over parameters owned by the optimizer.

        Used to select parameters used by the optimizer for gradient clipping.

        Args:
            optimizer: Target optimizer.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    @staticmethod
    def _model_train_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a trainig forward step. Compute model outputs and losses.

        Args:
            batch (Dict): [description]
            model (nn.Module): [description]
            criterion (nn.Module): [description]
            optimizer_idx (int, optional): [description]. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: [description]
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        # unwrap model in DDP training
        if hasattr(model, "module"):
            return model.module.train_step(*input_args)
        return model.train_step(*input_args)

    def _get_autocast_args(self, mixed_precision: bool, precision: str):
        device = "cpu"
        dtype = torch.get_autocast_cpu_dtype()
        if self.use_cuda:
            device = "cuda"
            dtype = torch.float32
            if mixed_precision:
                if precision == "fp16":
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    raise ValueError(f" ❗ Unknown precision {precision}")
        elif mixed_precision:
            dtype = torch.bfloat16
        return device, dtype

    def detach_loss_dict(
        self, loss_dict: Dict, step_optimizer: bool, optimizer_idx: int = None, grad_norm: float = None
    ):
        # detach losses for logging
        loss_dict_detached = self._detach_loss_dict(loss_dict)
        # loss_dict_detached["loss"] = loss_di`ct_detached["loss"] * float(self.grad_accum_steps)

        if optimizer_idx is not None:
            loss_dict_detached[f"loss_{optimizer_idx}"] = loss_dict_detached.pop("loss")
            if step_optimizer and grad_norm is not None:
                loss_dict_detached[f"grad_norm_{optimizer_idx}"] = grad_norm
        else:
            if step_optimizer and grad_norm is not None:
                loss_dict_detached["grad_norm"] = grad_norm
        return loss_dict_detached

    def _compute_loss(self, batch: Dict, model: nn.Module, criterion: nn.Module, config: Coqpit, optimizer_idx: int):
        device, dtype = self._get_autocast_args(config.mixed_precision, config.precision)
        with torch.autocast(device_type=device, dtype=dtype, enabled=config.mixed_precision):
            if optimizer_idx is not None:
                outputs, loss_dict = self._model_train_step(batch, model, criterion, optimizer_idx=optimizer_idx)
            else:
                outputs, loss_dict = self._model_train_step(batch, model, criterion)
        return outputs, loss_dict

    @staticmethod
    def _set_grad_clip_per_optimizer(config: Coqpit, optimizer_idx: int):
        # set gradient clipping threshold
        grad_clip = 0.0  # meaning no gradient clipping
        if "grad_clip" in config and config.grad_clip is not None:
            if optimizer_idx is not None:
                try:
                    grad_clip = config.grad_clip[optimizer_idx]
                except TypeError:
                    logger.info(" [!] You are using multiple optimizers but `grad_clip` is not a list.")
            else:
                grad_clip = config.grad_clip
        return grad_clip

    def _compute_grad_norm(self, optimizer: torch.optim.Optimizer):
        return torch.norm(torch.cat([param.grad.view(-1) for param in self.master_params(optimizer)], dim=0), p=2)

    def _grad_clipping(self, grad_clip: float, optimizer: torch.optim.Optimizer, scaler: "AMPScaler"):
        """Perform gradient clipping"""
        if grad_clip is not None and grad_clip > 0:
            if scaler:
                scaler.unscale_(optimizer)
            self.callbacks.before_gradient_clipping(self)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
        else:
            grad_norm = self._compute_grad_norm(optimizer)
        return grad_norm

    def optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: "AMPScaler",
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List, Dict],  # pylint: disable=protected-access
        config: Coqpit,
        optimizer_idx: int = None,
        step_optimizer: bool = True,
        num_optimizers: int = 1,
    ) -> Tuple[Dict, Dict, int]:
        """Perform a forward - backward pass and run the optimizer.

        Args:
            batch (Dict): Input batch. If
            model (nn.Module): Model for training. Defaults to None.
            optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
            scaler (AMPScaler): AMP scaler.
            criterion (nn.Module): Model's criterion.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler used by the optimizer.
            config (Coqpit): Model config.
            optimizer_idx (int, optional): Target optimizer being used. Defaults to None.
            step_optimizer (bool, optional): Whether step the optimizer. If False, gradients are accumulated and
                model parameters are not updated. Defaults to True.
            num_optimizers (int, optional): Number of optimizers. Defaults to 1.

        Raises:
            RuntimeError: When the loss is NaN.

        Returns:
            Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
        """

        step_start_time = time.time()

        # forward pass and loss computation
        outputs, loss_dict = self._compute_loss(
            batch=batch, model=model, criterion=criterion, config=config, optimizer_idx=optimizer_idx
        )

        # skip the rest if not outputs from the model
        if not loss_dict:
            step_time = time.time() - step_start_time
            return outputs, {}, step_time

        grad_clip = self._set_grad_clip_per_optimizer(config=config, optimizer_idx=optimizer_idx)
        # optimizer step
        grad_norm = 0
        update_lr_scheduler = True

        # callback
        self.callbacks.before_backward_pass(self, loss_dict)

        # accumulated gradients adjustment
        loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)

        if self.use_accelerate:
            with self.accelerator.accumulate(model):
                ctx_mgr = self.accelerator.autocast if config.mixed_precision else nullcontext
                with ctx_mgr():
                    self.accelerator.backward(loss_dict["loss"])
                    grad_norm = self._compute_grad_norm(optimizer)
                    if self.accelerator.sync_gradients and grad_clip is not None and grad_clip > 0:
                        self.accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    if not self.config.scheduler_after_epoch and not self.accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        else:
            if self.use_amp_scaler:
                if self.use_apex:
                    # TODO: verify AMP use for GAN training in TTS
                    # https://nvidia.github.io/apex/advanced.html?highlight=accumulate#backward-passes-with-multiple-optimizers
                    with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if step_optimizer:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=None)
                else:
                    # model optimizer step in mixed precision mode
                    scaler.scale(loss_dict["loss"]).backward()
                    # gradient accumulation
                    if step_optimizer:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=scaler)
                        scale_prev = scaler.get_scale()
                        scaler.step(optimizer)
                        # update the scaler at the end of all the optimizer steps
                        if optimizer_idx is None or (optimizer_idx + 1 == num_optimizers):
                            scaler.update()
                            loss_dict["amp_scaler"] = scaler.get_scale()  # for logging
                        update_lr_scheduler = scale_prev <= scaler.get_scale()
            else:
                # main model optimizer step
                loss_dict["loss"].backward()
                # gradient accumulation
                if step_optimizer:
                    self.callbacks.before_gradient_clipping(self)
                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
                    optimizer.step()

            # setup lr
            if (
                scheduler is not None
                and update_lr_scheduler
                and not self.config.scheduler_after_epoch
                and step_optimizer
            ):
                scheduler.step()

            # zero-out optimizer
            if step_optimizer:
                optimizer.zero_grad(set_to_none=True)

        # pytorch skips the step when the norm is 0. So ignore the norm value when it is NaN
        if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            grad_norm = 0

        step_time = time.time() - step_start_time

        # detach loss dict
        loss_dict_detached = self.detach_loss_dict(loss_dict, step_optimizer, optimizer_idx, grad_norm)
        return outputs, loss_dict_detached, step_time

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        """Perform a training step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
            step (int): Current step number in this epoch.
            loader_start_time (float): The time when the data loading is started. Needed for logging.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        self.callbacks.on_train_step_start(self)
        # format data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        # conteainers to hold model outputs and losses for each optimizer.
        outputs_per_optimizer = None
        loss_dict = {}

        # OPTIMIZATION
        if isimplemented(self.model, "optimize"):  # pylint: disable=too-many-nested-blocks
            # custom optimize for the model
            step_time = time.time()
            device, dtype = self._get_autocast_args(self.config.mixed_precision, self.config.precision)
            with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
                outputs, loss_dict_new = self.model.optimize(
                    batch,
                    self,
                )
            step_time = time.time() - step_time
            # If None, skip the step
            if outputs is None:
                return None, None
            # TODO: find a way to log grad_norm for custom optimize
            loss_dict_new = self.detach_loss_dict(loss_dict_new, True, None, None)
            loss_dict.update(loss_dict_new)
        else:
            # gradient accumulation
            # TODO: grad accumulation for each optimizer
            step_optimizer = True
            if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
                step_optimizer = False

            if not isinstance(self.optimizer, list):
                # auto training with a single optimizer
                outputs, loss_dict_new, step_time = self.optimize(
                    batch,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.criterion,
                    self.scheduler,
                    self.config,
                    step_optimizer=step_optimizer,
                    num_optimizers=1,
                )
                loss_dict.update(loss_dict_new)
            else:
                # auto training with multiple optimizers (e.g. GAN)
                outputs_per_optimizer = [None] * len(self.optimizer)
                total_step_time = 0
                for idx, optimizer in enumerate(self.optimizer):
                    criterion = self.criterion
                    # scaler = self.scaler[idx] if self.use_amp_scaler else None
                    scaler = self.scaler
                    scheduler = None
                    if self.scheduler is not None:
                        scheduler = self.scheduler[idx]
                    outputs, loss_dict_new, step_time = self.optimize(
                        batch,
                        self.model,
                        optimizer,
                        scaler,
                        criterion,
                        scheduler,
                        self.config,
                        idx,
                        step_optimizer=step_optimizer,
                        num_optimizers=len(self.optimizer),
                    )
                    # skip the rest if the model returns None
                    total_step_time += step_time
                    outputs_per_optimizer[idx] = outputs
                    # merge loss_dicts from each optimizer
                    # rename duplicates with the optimizer idx
                    # if None, model skipped this optimizer
                    if loss_dict_new is not None:
                        for k, v in loss_dict_new.items():
                            if k in loss_dict:
                                loss_dict[f"{k}-{idx}"] = v
                            else:
                                loss_dict[k] = v
                    step_time = total_step_time

                outputs = outputs_per_optimizer

                # clear any pesky gradients after gradient accumulation
                if step_optimizer:
                    self.model.zero_grad(set_to_none=True)

        # update avg runtime stats
        keep_avg_update = {}
        keep_avg_update["avg_loader_time"] = loader_time
        keep_avg_update["avg_step_time"] = step_time
        self.keep_avg_train.update_values(keep_avg_update)

        # update avg loss stats
        update_eval_values = {}
        for key, value in loss_dict.items():
            update_eval_values["avg_" + key] = value
        self.keep_avg_train.update_values(update_eval_values)

        # print training progress
        if self.total_steps_done % self.config.print_step == 0:
            # log learning rates
            lrs = {}
            if isinstance(self.optimizer, list):
                for idx, optimizer in enumerate(self.optimizer):
                    current_lr = self.optimizer[idx].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{idx}": current_lr})
            elif isinstance(self.optimizer, dict):
                for key, optimizer in self.optimizer.items():
                    current_lr = self.optimizer[key].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{key}": current_lr})
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                lrs = {"current_lr": current_lr}

            # log run-time stats
            loss_dict.update(lrs)
            loss_dict.update(
                {
                    "step_time": round(step_time, 4),
                    "loader_time": round(loader_time, 4),
                }
            )
            self.c_logger.print_train_step(
                batch_n_steps,
                step,
                self.total_steps_done,
                loss_dict,
                self.keep_avg_train.avg_values,
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load and don't log every step
            if self.total_steps_done % self.config.plot_step == 0:
                self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
            if self.total_steps_done % self.config.save_step == 0 and self.total_steps_done != 0:
                if self.config.save_checkpoints:
                    # checkpoint the model
                    self.save_checkpoint()

            if self.total_steps_done % self.config.log_model_step == 0:
                # log checkpoint as artifact
                self.update_training_dashboard_logger(batch=batch, outputs=outputs)

            self.dashboard_logger.flush()

        self.total_steps_done += 1
        self.callbacks.on_train_step_end(self)
        return outputs, loss_dict

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        # initialize the data loader
        if self.train_loader is None:
            self.train_loader = self.get_train_dataloader(
                self.training_assets,
                self.train_samples,
                verbose=True,
            )
            self.train_loader = self.prepare_accelerate_loader(self.train_loader)
        # set model to training mode
        torch.set_grad_enabled(True)
        if self.num_gpus > 1:
            self.model.module.train()
        else:
            self.model.train()
        epoch_start_time = time.time()

        self.callbacks.on_train_epoch_start(self)

        self.c_logger.print_train_start()
        loader_start_time = time.time()
        # TRAINING EPOCH -> iterate over the training samples
        batch_num_steps = len(self.train_loader)
        for cur_step, batch in enumerate(self.train_loader):
            outputs, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
            if outputs is None:
                logger.info(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
                continue
            del outputs
            loader_start_time = time.time()

            # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
            if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                self.eval_epoch()
                if self.num_gpus > 1:
                    self.model.module.train()
                else:
                    self.model.train()
                torch.set_grad_enabled(True)


        epoch_time = time.time() - epoch_start_time
        self.callbacks.on_train_epoch_end(self)

        # scheduler step
        if self.scheduler is not None and self.config.scheduler_after_epoch:
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    if scheduler is not None:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(self._pick_target_avg_loss(self.keep_avg_eval) if not None else 999)
                        else:
                            scheduler.step()
            elif isinstance(self.scheduler, dict):  # only with `model.optimize()``
                for scheduler in self.scheduler.values():
                    if scheduler is not None:
                        scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self._pick_target_avg_loss(self.keep_avg_eval) if not None else 999)
            else:
                self.scheduler.step()
        # plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.model_param_stats:
                self.dashboard_logger.model_weights(self.model, self.total_steps_done)
        torch.cuda.empty_cache()

    #######################
    # EVAL FUNCTIONS
    #######################

    def _model_eval_step(
        self, batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (nn.Module): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args = [batch, criterion]

        if isimplemented(model, "optimize"):
            if hasattr(model, "module"):
                return model.module.eval_step(batch, self)
            return model.eval_step(batch, self)

        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        if hasattr(model, "module"):
            return model.module.eval_step(*input_args)

        return model.eval_step(*input_args)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        with torch.no_grad():
            outputs = []
            loss_dict = {}
            if not isinstance(self.optimizer, list) or isimplemented(self.model, "optimize"):
                outputs, loss_dict = self._model_eval_step(batch, self.model, self.criterion)
                if outputs is None:
                    return None, None
            else:
                outputs = [None] * len(self.optimizer)
                for idx, _ in enumerate(self.optimizer):
                    criterion = self.criterion
                    outputs_, loss_dict_new = self._model_eval_step(batch, self.model, criterion, idx)
                    if outputs_ is None:
                        return None, None
                    outputs[idx] = outputs_

                    if loss_dict_new:
                        loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
                        loss_dict.update(loss_dict_new)

            loss_dict = self._detach_loss_dict(loss_dict)

            # update avg stats
            update_eval_values = {}
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            self.keep_avg_eval.update_values(update_eval_values)
            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)

        return outputs, loss_dict

    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""

        # initialize it when eval_epoch is called alone.
        self.keep_avg_eval = KeepAverage() if self.keep_avg_eval is None else self.keep_avg_eval

        if self.eval_loader is None:
            self.eval_loader = (
                self.get_eval_dataloader(
                    self.training_assets,
                    self.eval_samples,
                    verbose=True,
                )
                if self.config.run_eval
                else None
            )

        torch.set_grad_enabled(False)
        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        outputs = None
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs_, _ = self.eval_step(batch, cur_step)
            if outputs_ is None:
                logger.info(" [!] `eval_step()` retuned `None` outputs. Skipping evaluation step.")
                continue
            outputs = outputs_
            loader_start_time = time.time()
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0 and outputs is not None:
            if hasattr(self.model, "module") and isimplemented(self.model.module, "eval_log"):
                self.model.module.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif isimplemented(self.model, "eval_log"):
                self.model.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
        torch.cuda.empty_cache()

    ##################################
    # TESTING
    ##################################
    def test_run(self) -> None:
        """Run model test.

        Test run is expected to pass over test samples and produce logging artifacts.

        If ```model.test_run()``` is defined, it will be called and it is expected to set and execute everything
        in the model.

        Else if  ```mode.test()``` is defined, it will be called and it takes an test data loader as an argument
        and iterate over it.
        """
        self.model.eval()
        test_outputs = None
        if isimplemented(self.model, "test_run") or (
            self.num_gpus > 1 and isimplemented(self.model.module, "test_run")
        ):
            # handle everything in ```model.test_run()`
            if self.num_gpus > 1:
                test_outputs = self.model.module.test_run(self.training_assets)
            else:
                test_outputs = self.model.test_run(self.training_assets)
        elif isimplemented(self.model, "test") or (self.num_gpus > 1 and isimplemented(self.model.module, "test")):
            self.test_loader = self.get_test_dataloader(
                self.training_assets,
                self.test_samples if self.test_samples else self.eval_samples,
                verbose=True,
            )
            # use test_loader to load test samples
            if self.num_gpus > 1:
                test_outputs = self.model.module.test(self.training_assets, self.test_loader, None)
            else:
                test_outputs = self.model.test(self.training_assets, self.test_loader, None)
        if isimplemented(self.model, "test_log") or (
            self.num_gpus > 1 and isimplemented(self.model.module, "test_log")
        ):
            if self.num_gpus > 1:
                self.model.module.test_log(
                    test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done
                )
            else:
                self.model.test_log(test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done)

    def _restore_best_loss(self):
        """Restore the best loss from the args.best_path if provided else
        from the model (`args.continue_path`) used for resuming the training"""
        if self.args.continue_path and (self.restore_step != 0 or self.args.best_path):
            logger.info(" > Restoring best loss from %s ...", os.path.basename(self.args.best_path))
            ch = load_fsspec(self.args.restore_path, map_location="cpu")
            if "model_loss" in ch:
                if isinstance(ch["model_loss"], dict):
                    self.best_loss = ch["model_loss"]
                # For backwards-compatibility:
                elif isinstance(ch["model_loss"], float):
                    if self.config.run_eval:
                        self.best_loss = {"train_loss": None, "eval_loss": ch["model_loss"]}
                    else:
                        self.best_loss = {"train_loss": ch["model_loss"], "eval_loss": None}
            logger.info(" > Starting with loaded last best loss %s", self.best_loss)

    def test(self, model=None, test_samples=None) -> None:
        """Run evaluation steps on the test data split. You can either provide the model and the test samples
        explicitly or the trainer use values from the initialization.

        Args:
            model (nn.Module, optional): Model to use for testing. If None, use the model given in the initialization.
                Defaults to None.

            test_samples (List[str], optional): List of test samples to use for testing. If None, use the test samples
                given in the initialization. Defaults to None.
        """

        logger.info(" > USING TEST SET...")
        self.keep_avg_eval = KeepAverage()

        if model is not None:
            self.model = model

        eval_samples_cache = self.eval_samples
        if test_samples is not None:
            self.eval_samples = test_samples
        else:
            self.eval_samples = self.test_samples

        self.eval_epoch()
        self.c_logger.print_epoch_end(self.epochs_done, self.keep_avg_eval.avg_values)
        self.eval_samples = eval_samples_cache

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.keep_avg_eval = KeepAverage()
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()

            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False

    def fit_with_largest_batch_size(self, starting_batch_size=2048) -> None:
        cuda_meminfo()
        bs = starting_batch_size
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                gc.collect()
                torch.cuda.empty_cache()
                self.config.batch_size = bs
                logger.info(" > current batch size: %i", self.config.batch_size)
                self._fit()
            except RuntimeError as exception:
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as exception:  # pylint: disable=broad-except
                # catches the torch.cuda.OutOfMemoryError
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            else:
                break

    def fit(self) -> None:
        """Where the ✨️magic✨️ happens..."""
        try:
            self._fit()
            if self.args.rank == 0:
                self.dashboard_logger.finish()
        except KeyboardInterrupt:
            logger.info(" > Keyboard interrupt detected.")
            if self.config.save_on_interrupt:
                logger.info(" > Saving model before exiting...")
                # save the model on keyboard interrupt
                self.save_checkpoint()
                # update the training dashboard logger
                self.update_training_dashboard_logger()
            # call the keyboard interrupt callback
            self.callbacks.on_keyboard_interrupt(self)
            # if the output folder is empty remove the run.
            remove_experiment_folder(self.output_path)
            # clear the DDP processes
            if self.num_gpus > 1:
                dist.destroy_process_group()
            # finish the wandb run and sync data
            if self.args.rank == 0:
                self.dashboard_logger.finish()
            # stop without error signal
            try:
                sys.exit(1)
            except SystemExit:
                os._exit(1)  # pylint: disable=protected-access
        except BaseException:  # pylint: disable=broad-except
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)

    def profile_fit(self, torch_profiler, epochs=None, small_run=None):
        """Run training under the torch profiler.

        Example::
            Run torch profiler to profile CPU, GPU and memory usage with Tensorboard logging.

            >>> import torch
            >>> profiler = torch.profiler.profile(
            >>>        activities=[
            >>>         torch.profiler.ProfilerActivity.CPU,
            >>>         torch.profiler.ProfilerActivity.CUDA,
            >>>     ],
            >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
            >>>     record_shapes=True,
            >>>     profile_memory=True,
            >>>     with_stack=True,
            >>> )
            >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
        """
        self.dashboard_logger = DummyLogger()
        # train the model for a custom number of epochs
        if epochs:
            self.config.epocshs = epochs
        # use a smaller set of training samples for profiling
        if small_run:
            self.setup_small_run(small_run)
        # run profiler
        self.config.run_eval = False
        self.config.test_delay_epochs = 9999999
        self.config.epochs = epochs
        # set a callback to progress the profiler
        self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
            lambda trainer: trainer.torch_profiler.step()
        ]
        # set the profiler to access in the Trainer
        self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init
        # set logger output for Tensorboard
        # self.torch_profiler.on_trace_ready = torch.profiler.tensorboard_trace_handler(self.output_path)
        self.torch_profiler.start()
        self.fit()
        self.torch_profiler.stop()
        return self.torch_profiler

    @rank_zero_only
    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""

        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        # save the model and update the best_loss
        self.best_loss = save_best_model(
            {"train_loss": train_loss, "eval_loss": eval_loss},
            self.best_loss,
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            keep_all_best=self.config.save_all_best,
            keep_after=self.config.save_best_after,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def save_checkpoint(self) -> None:
        """Save the current model checkpoint."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        save_checkpoint(
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
            save_n_checkpoints=self.config.save_n_checkpoints,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def update_training_dashboard_logger(self, batch=None, outputs=None):
        aliases = [
            f"epoch-{self.epochs_done}",
            f"step-{self.total_steps_done}",
        ]
        self.dashboard_logger.add_artifact(
            file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
        )

        # training visualizations
        if batch is not None and outputs is not None:
            if hasattr(self.model, "module") and isimplemented(self.model.module, "train_log"):
                self.model.module.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif isimplemented(self.model, "train_log"):
                self.model.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )

    #####################
    # GET FUNCTIONS
    #####################

    @staticmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> Union[torch.optim.Optimizer, List]:
        """Receive the optimizer from the model if model implements `get_optimizer()` else
        check the optimizer parameters in the config and try initiating the optimizer.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
        """
        optimizer = None
        if isimplemented(model, "get_optimizer"):
            try:
                optimizer = model.get_optimizer()
            except NotImplementedError:
                optimizer = None
        if optimizer is None:
            optimizer_name = config.optimizer
            optimizer_params = {} if config.optimizer_params is None else config.optimizer_params
            return get_optimizer(optimizer_name, optimizer_params, config.lr, model)
        return optimizer

    @staticmethod
    def get_lr(model: nn.Module, config: Coqpit) -> Union[float, List[float]]:
        """Set the initial learning rate by the model if model implements `get_lr()` else try setting the learning rate
        fromthe config.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
        """
        lr = None
        if isimplemented(model, "get_lr"):
            try:
                lr = model.get_lr()
            except NotImplementedError:
                lr = None
        if lr is None:
            lr = config.lr
        return lr

    @staticmethod
    def get_scheduler(
        model: nn.Module, config: Coqpit, optimizer: Union[torch.optim.Optimizer, List, Dict]
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, List]:  # pylint: disable=protected-access
        """Receive the scheduler from the model if model implements `get_scheduler()` else
        check the config and try initiating the scheduler.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List, Dict]: A scheduler or a list of schedulers, one for each optimizer.
        """
        scheduler = None
        if isimplemented(model, "get_scheduler"):
            try:
                scheduler = model.get_scheduler(optimizer)
            except NotImplementedError:
                scheduler = None
            if isinstance(scheduler, dict) and not isimplemented(model, "optimize"):
                raise ValueError(
                    " [!] Dictionary of schedulers are only supported with the manual optimization `model.optimize()`."
                )
        if scheduler is None:
            lr_scheduler = config.lr_scheduler
            lr_scheduler_params = config.lr_scheduler_params
            return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)
        return scheduler

    @staticmethod
    def restore_scheduler(
        scheduler: Union["Scheduler", List, Dict], args: Coqpit, config: Coqpit, restore_epoch: int, restore_step: int
    ) -> Union["Scheduler", List]:
        """Restore scheduler wrt restored model."""
        if scheduler is not None:  # pylint: disable=too-many-nested-blocks
            if args.continue_path:
                if isinstance(scheduler, list):
                    for s in scheduler:
                        if s is not None:
                            if config.scheduler_after_epoch:
                                s.last_epoch = restore_epoch
                            else:
                                s.last_epoch = restore_step
                elif isinstance(scheduler, dict):
                    for s in scheduler.values():
                        if s is not None:
                            if config.scheduler_after_epoch:
                                s.last_epoch = restore_epoch
                            else:
                                s.last_epoch = restore_step
                else:
                    if config.scheduler_after_epoch:
                        scheduler.last_epoch = restore_epoch
                    else:
                        scheduler.last_epoch = restore_step
        return scheduler

    @staticmethod
    def get_criterion(model: nn.Module) -> nn.Module:
        """Receive the criterion from the model. Model must implement `get_criterion()`.

        Args:
            model (nn.Module): Training model.

        Returns:
            nn.Module: Criterion layer.
        """
        criterion = None
        criterion = model.get_criterion()
        return criterion

    ####################
    # HELPER FUNCTIONS
    ####################

    @staticmethod
    def _detach_loss_dict(loss_dict: Dict) -> Dict:
        """Detach loss values from autograp.

        Args:
            loss_dict (Dict): losses.

        Returns:
            Dict: losses detached from autograph.
        """
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.detach().cpu().item()
        return loss_dict_detached

    def _pick_target_avg_loss(self, keep_avg_target: KeepAverage) -> Dict:
        """Pick the target loss to compare models"""

        # if the keep_avg_target is None or empty return None
        if keep_avg_target is None or len(list(keep_avg_target.avg_values.keys())) == 0:
            return None

        target_avg_loss = None
        # return if target loss defined in the model config
        # if not available in Dict use loss_1 as by default loss
        if "target_loss" in self.config and self.config.target_loss:
            if f"avg_{self.config.target_loss}" in keep_avg_target.avg_values.keys():
                return keep_avg_target[f"avg_{self.config.target_loss}"]

            raise ValueError(
                " [!] Target loss not found in the keep_avg_target. You might be exiting the training loop before it is computed or set the target_loss in the model config incorrectly."
            )

        # take the average of loss_{optimizer_idx} as the target loss when there are multiple optimizers
        if isinstance(self.optimizer, list):
            target_avg_loss = 0
            for idx in range(len(self.optimizer)):
                if f"avg_loss_{idx}" in keep_avg_target.avg_values:
                    target_avg_loss += keep_avg_target[f"avg_loss_{idx}"]
            target_avg_loss /= len(self.optimizer)
        else:
            target_avg_loss = keep_avg_target.avg_values.get("avg_loss", 0)
        return target_avg_loss

    def _setup_logger_config(self, log_file: str) -> None:
        """Set up the logger based on the process rank in DDP."""

        logger_new = logging.getLogger("trainer")
        handler = logging.FileHandler(log_file, mode="a")
        fmt = logging.Formatter("")
        handler.setFormatter(fmt)
        logger_new.addHandler(handler)

        # only log to a file if rank > 0 in DDP
        if self.args.rank > 0:
            logger_new.handlers = [h for h in logger_new.handlers if not isinstance(h, logging.StreamHandler)]

    @staticmethod
    def _is_apex_available() -> bool:
        """Check if Nvidia's APEX is available."""
        return importlib.util.find_spec("apex") is not None
