# ViDLU: Vision Deep Learning Utilities

A deep learning framework for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/). Many parts are experimental or incomplete.

[![Build Status](https://github.com/Ivan1248/Vidlu/workflows/build/badge.svg)](https://github.com/Ivan1248/Vidlu/actions)
[![codecov](https://codecov.io/gh/Ivan1248/Vidlu/branch/master/graph/badge.svg)](https://codecov.io/gh/Ivan1248/Vidlu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7f89c65e677f490bab26c0e5c7cae116)](https://www.codacy.com/manual/Ivan1248/Vidlu?utm_source=github.com&utm_medium=referral&utm_content=Ivan1248/Vidlu&utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/vidlu/badge/?version=latest)](https://vidlu.readthedocs.io/en/latest/?badge=latest)

This repository contains

1. an experimental machine learning framework, mostly based on PyTorch,
2. a set of datasets, models, training configurations (as part of the framework), various related algorithms, and
3. a set of scripts for running experiments.

## Setup

**Note:** `pip install` is usually not recommended outside a virtual environment. To install the required packages, it is probably best to use the package manager that you usually use.

**No installation.**
You can make a local copy and install dependencies with

```sh
git clone https://github.com/Ivan1248/vidlu.git
cd vidlu
pip install -r requirements.txt
```

**Pip installation.**
Alternatively, you can install the package with

```sh
pip install git+https://github.com/Ivan1248/vidlu
```

<!--
or, if there is a local copy of the repository,
```sh
pip install .
```
-->

## Main scripts

The "scripts" directory contains scripts that use [the framework](#the-framework). `run.py` can run experiments and `dirs.py` contains directory paths.

### Directory configuration

`scripts/dirs.py` searches for and stores directory paths for datasets, cache, results and other data in the following variables:
-   `datasets: list[Path]` can point to multiple directories, each of which can contain dataset directories.
-   `cache: Path` points to a directory for caching data.
-   `pretrained: Path` points to a directory for pre-trained parameters.
-   `experiments: Path` points to a directory for experiment results. The directory `saved_states = experiments / "states"` is automatically created for storing intermediate and complete training states.

It might be easiest to create the following directory structure. Symbolic links can be useful.

```sh
<ancestor>
├─ .../vidlu/scripts/dirs.py
└─ data
   ├─ cache
   ├─ datasets
   ├─ experiments  # subdirectories created automatically
   │  └─ states
   └─ pretrained
```

The "data" directory can be created in the user home directory by running

```sh
mkdir ~/data ~/data/datasets ~/data/cache ~/data/experiments ~/data/pretrained
```

"data" is found automatically if its parent directory is also an ancestor of `dirs.py`. Otherwise, the environment variable `VIDLU_DATA` should point to the "data" directory.

The "cache" directory should preferably be on an SSD. "datasets" and other directories, on a slower disk. Data from "datasets" is not accessed after being cached.

Alternatively, the paths can be defined individually through multiple environment variables: `VIDLU_DATASETS`, `VIDLU_CACHE`, `VIDLU_PRETRAINED`, and `VIDLU_EXPERIMENTS`.

### Running experiments

`scripts/run.py` is a general script for running experiments.

The `train` command is chosen by running `python run.py train ...`. It creates an `Experiment` instance from command line arguments and directory paths from `dirs.py`. The `Experiment` constructor creates a `Trainer` instance using factories from `vidlu.factories`. The `train` command runs evaluation and training. Interrupted or completed experiments can be continued/reevaluated using the `--resume` (`-r`) argument. The command can have the following structure:

```sh
run.py train DATA INPUT_ADAPTER MODEL TRAINER [-h] [--params PARAMS] [--metrics METRICS] [-e EXPERIMENT_SUFFIX] [-r [{strict,?,best,restart}]]
```

There is also a `test` command that accepts almost the same arguments and can be used for standard evaluation or running a custom procedure that can optionally accept the `Experiment` instance as one of its arguments.

`scripts/train_cifar.py` is a specific example where it is easier to tell what is happening.
Running `python train_cifar.py` is equivalent to running the following training with modified hyperparameters.

```sh
python run.py train \
    "Cifar10{trainval,test}" "id" \  # data
    "models.ResNetV1,backbone_f=t(depth=18,small_input=True,block_f=t(norm_f=None))" \  # model
    "ct.resnet_cifar,lr_scheduler_f=ConstLR,epoch_count=50,jitter=None"  # training
```

Note the example has some changes with respect to the default CIFAR-10 configuration: disabled batchnorm, constant learning rate, 50 epochs, disabled jittering.

## The framework

Some of the main packages in the framework are `vidlu.data`, `vidlu.modules`, `vidlu.models`, `vidlu.training`, `vidlu.metrics`, `vidlu.factories`, `vidlu.configs`, `vidlu.utils` and `vidlu.experiment`.

Most of the code here is generic except for concrete datasets in `vidlu.data.datasets`, hyperparameter configurations and other data in `vidlu.configs`, concrete models in `vidlu.models`, some modules in `vidlu.modules.components`, and `vidlu.experiments`, which applies the framework for training and evaluation.

### Data

`vidlu.data` defines types `Record`, `Dataset` and PyTorch `DataLoader`-based types. There are also many concrete datasets in `vidlu.data.datasets`.

`Record` is an ordered key-value mapping that supports lazy evaluation of values. It can be useful when not all fields of dataset examples need to be loaded.

`Dataset` is the base dataset class. It has a set of useful methods for manipulation and caching (advanced indexing, concatenation, `map`, `filter`, ...).

`DataLoader` inherits `DataLoader` from PyTorch and changes its `default_collate` so that it supports elements of type `Record`.

### Modules (model components) and models

`vidlu.modules` contains implementations of various modules and functions (`elements`, `components`, `heads`, `losses`) and useful procedures for debugging, extending and manipulating modules.
The modules (inheriting `Module`) support shape inference like in e.g. [MXNet](http://mxnet.incubator.apache.org/) and [MagNet](https://github.com/MagNet-DL/magnet) (an initial run in necessary for initialization).

`try_get_module_name_from_call_stack` enables getting the name of the current module.

`Seq` is an alternative for `Sequential` which supports splitting, joining, and other things. Many modules are based on it. `deep-split` (accepting a path to some inner module) and `deep_join` can work on composite models that are designed based on `Sequential`.

`with_intermediate_outputs` can be used for extracting intrmediate outputs without changing the module. It uses `register_forward_hook` (and thus requires appropriately designed models).

For many elementary modules which can be invertible, the `inverse` property returns its inverse module. The inverse is defined either via a `make_inverse` or `inverse_forward`. A `Seq` which consists of only invertible modules (like `Identity`, `Permute`, `FactorReshape`, ...) is automatically invertible. Without a change in the interface, invertible modules also support optional computation and propagation of the logarithm of volume change necessary for normalizing flows.

`vidlu.modules.pert_models` defines parametrized perturbation models that use independent parameters for each input in the mini-batch or mixe data between inputs.

`vidlu.modules.losses` contains loss functions.

Composite modules are designed to be "deeply" configurable: arguments of arguments that are factories/constructors for child modules can be modified. Names of such factory arguments usually end with `_f`. If a default argument is a function, its arguments can be accessed and modified using `vidlu.utils.func`, which relies on `inspect.signature` and `functools.partial`. `vidlu.utils.func` defines tree data structures and procedures that enable eays modification of deeply nested arguments.

`vidlu.models` contains implementations of some models. Model classes are mostly wrappers around more general modules defined in `vidlu.modules.components` and heads defined in `vidlu.modules.heads`. They also perform initialization of parameters. Some implementad architectures are ResNet-v1, ResNet-v2, Wide ResNet, DenseNet, i-RevNet, SwiftNet, Ladder-DenseNet<sup>[1](#fn1)</sup>.

<a name="fn1">1</a>: There might be some unintended differences to the original code.

### Training

`vidlu.training` defines procedural machine learning algorithm components.

`EpochLoop` (based on `Engine` from [Ignite](https://github.com/pytorch/ignite)) is used for running training or evaluation loops. The iteration step procedure is an argument to the constructor. It raises events before and after the loop and the iteration step.

`Trainer` defines a full machine learning algorithm. It has `train` and `eval` methods. Some of its more important attributes (components) are: `model`, `eval_batch_size` (E), `metrics` (E), `eval_step` (E), `loss` (L),  `batch_size` (L), `jitter` (L), `train_step` (L), `extensions` (L), `epoch_count` (O), `optimizer` (O), `lr_scheduler` (O), `data_loader_f` (D). E denotes evaluation components, which do not affect training, L learning components, and O learning components mostly related to optimization.

`CheckpointManager` is used for storing and loading the state of a trainer (and some other things) in the file system.

`vidlu.training.steps` defines training and evaluation steps. Instances of step classes (inheriting `BaseStep`) have a `__call__` method that accepts a `Trainer` instance and a data batch. Training steps can be stateful and might need to define `state_dict` and `load_state_dict` methods. There are steps of different supervised, adversarial, semi-supervised, normalizing flow, and some hybrid algorithms.

### Factories

`vidlu.factories` contains the factories that can create model, data, learning, and evaluation components from strings representing Python expressions (which can be provided through the command line). Note that it uses Python's `eval` extensively.

`get_prepared_data_for_trainer` accepts a string containing the names of the datasets (with subset names) and code of an arbitrary transformations applied to them (using `Dataset`'s methods, `vidlu.data.utils.dataset_ops`, `vidlu.transforms`, `torchvision.transforms`...). It also requires `datasets_dir` and `cache_dir`, which represent paths to root directories for datasets and cache. It returns a sequence of `Dataset` instances with the transformations applied. The returned `Dataset` instances also convert images and labels to PyTorch `Tensor` instances. They also scale images to range [0..1] and transpose them to the CHW format.

`get_model` (among other arguments) accepts a string containing the name of the model and a list of arguments. The list of arguments is separated by a comma from the model name. The model name should be either (1) a symbol from `vidlu.models`, (2) a symbol reachable through module located in paths listed in the `VIDLU_EXTENSIONS` environment variable, (3) a Pytorch Hub identifier that can be given to `torch.hub.load`. The list of arguments can contain argument trees constructed by nesting calls to the `ArgTree` constructor (`t` is a short alias), or other appropriate `UpdaTree` classes. Some other arguments that `get_model` accepts are `input_adapter_str` (a string defining input pre-processing), `prep_dataset` (prepared dataset used for getting possible problem-specific information and inputs for model initialization) and `device`.

`get_trainer` accepts a string representing an argument list for the `TrainingConfig` constructor and a model. Keyword arguments can be defined as trees (appropriate instances of `UpdaTree` from `vidlu.utils.func`) that are used to update (without mutation) `TrainingConfig` elements and objects within.

Custom modules can be made available for use in string expressions using [extensions](#extensions).

### Training configurations

Training hyperparameter configurations can be defined using the `TrainerConfig` class from `vidlu.configs.training`. The `TrainingConfig` constructor accepts 0 or more `TrainingConfig` instances as positional arguments and keyword arguments that correspond to parameters of the `Trainer` constructor. New configurations can be created by extending updating previously defined ones.

Optimizer configurations can be defined using `OptimizerMaker`, which stores all optimizer information while being decoupled from model parameters. In contrast to optimizers from `torch.optim` and `vidlu.optim` it stores module names instead of parameters. An `optimizer` instance can be created by calling an instance of `OptimizerMaker` with the model as the argument.

`vidlu.configs.training` contains many examples of configurations for things such as classification, semantic segmentation, adversarial training, semi-supervised learning, invertible models, ...

### Experiments

`vidlu.experiment` defines a program for creating and running experiments. It uses `vidlu.factories` to create a `Trainer`, it defines training and evaluation loop actions such as evaluation of performance metrics from `vidlu.metrics`, printing, logging, checkpoint management, user interaction (command execution and training/evaluation step output inspection), and training time estimation.

### Extensions

Vidlu enables extensions using the [_naming convention_ approach](https://packaging.python.org/guides/creating-and-discovering-plugins/#using-naming-convention). This means that installed packages or other packages found in directories in the `PYTHONPATH` environment variable with names prefixed with "vidlu\_" are loaded and made available in the `extensions` dictionary in the `vidlu.extensions` module, but the prefix is removed. For example, if the name of the package is `vidlu\_my_ext`, it will have the name `my_ext` in the `extensions` dictionary.

Extensions are directly available for expression arguments for [factories in `vidlu.factories`](#factories). For example, the code should work if `MyStep` and `MyModel` are defined in the extension `my_ext`:
```python
from torch import nn
from vidlu.factories import get_trainer

model = vidlu.extensions.extensions['my_ext'].MyModel()
trainer = get_trainer("ct.supervised_cifar, training_step=my_ext.MyStep, eval_step=None")
```

### Commonly used utilities

In many places in the code, some parameter names end with `_f`.
This means that the argument is not a final object, but a factory (hence `_f`). E.g. `block_f()` should produce a `block` instance. This is to allow more flexibility while keeping signatures short. Here the combination of such a design with `ArgTree` and `tree_partial` (analogue of `functools.partial`) enables flexible functional modification of any set of parameters of nested functions.

```py
from functools import partial as p
from vidlu.utils.func import tree_partial, ArgTree as t

def make_swallow(type='european', ...): ...

def make_flock(load=None, swallow_f=make_swallow, ...): ...

def eu_deliver(dest, flock_f=make_flock):
    ...
    flock = flock_f(...)
    ...

au_deliver_t = tree_partial(eu_deliver, flock_f=t(load='coconut', swallow_f=t(type='african')))
au_deliver_p = p(eu_deliver, flock_f=p(make_flock, load='coconut', swallow_f=p(make_swallow, type='african')))
dest = 'Caerbannog'
assert au_deliver_t(dest) == au_deliver_p(dest)
```

<!--
instead of

```
def foo(..., bar_args, baz_args):
    make_some_bar(..., **bar_args, **baz_args)
    ...

def make_some_bar(..., **baz_args):
    make_some_baz(..., **baz_args)
    ...
    
def make_some_baz(..., swallow_type='european'):
    ...

foo(baz_args=dict(swallow_type='african'))
```
-->

<!--

## Things (to be) done (currently not updated)

-   data  
    -   [x] easy data processing (transformations, splitting, joining)  
    -   [x] data caching in RAM and HDD  
    -   [x] lazy data loading and processing (e.g. if segmentation labels are not requested, they are not loaded)  
    -   [x] many datasets (data loaders)  
-   modules  
    -   [x] modules with shape inference (like in e.g. [MXNet](http://mxnet.incubator.apache.org/) and [MagNet](https://github.com/MagNet-DL/magnet)) -- an initial run in necessary for initialization  
    -   [x] procedures for getting module names (paths) from the call stack and extracting intermediate layer outputs through their names  
    -   [x] high generality/customizability of components and models (deep arguments) (not completely satisfactory)  
    -   [x] deep module splitting and joining (implemented for `vidlu.modules.Sequential`)  
    -   [x] extraction of intermediate outputs without changing the module (`vidlu.modules.with_intermediate_outputs`, using `register_forward_hook`)    
    -   [x] For many elementary modules which can be invertible, the `inverse` property returns its inverse module. The inverse is defined either via a `make_inverse` or `inverse_forward`. A `Seq` which consists of only invertible modules (like `Identity`, `Permute`, `FactorReshape`, ...) is automatically invertible.  
    -   [x] perturbation models with parameters with batch dimension  
-   models (convolutional) with training algorithms  
    -   classification  
        -   [x] ResNet, WRN, DenseNet  
        -   [x] iRevNet  
        -   [ ] RevNet, VGG  
        -   [x] other: BagNet  
    -   semantic segmentation  
        -   [x] basic segmentation head and loss  
        -   [x] SwiftNet  
        -   [ ] Ladder DenseNet  
    -   [ ] stochastic model parts  
    -   [ ] variational inference  
    -   [ ] VAE  
    -   [ ] flow generative models  
-   training  
    -   [x] training disentangled from the model  
    -   [x] customizable configurations  
    -   [x] logging, checkpoints, resuming interrupted training  
    -   [x] pretrained parameter loading: ResNet, DenseNet  
    -   [x] supervised training  
    -   [x] adversarial training  
    -   [x] semi-supervised training (VAT)  
    -   [ ] GAN training  
    -   [ ] natural gradient variational inference  
-   adversarial attacks  
    -   [x] single step gradient  
    -   [x] PGD/BIM  
    -   [x] PGD: stop on success  
    -   [x] PGD: support for "adversarial training for free"  
    -   [x] VAT attack  
    -   [x] perturbation models  
    -   [x] support for optimizers  
    -   [ ] CW  
    -   [ ] DDN (arXiv:1811.09600)  
    -   [ ] Frank-Wolfe constrained optimization  

In many places in the code some parameter names end with "_f". 
This means that the argument is not a final object but a factory (hence "_f") 
that produces an object, e.g. "backbone_f()" should produce a "backbone". This 
is to allow more flexibility e.g.

```
def foo(..., bar_f=make_some_bar):
    ...

def make_some_bar(..., baz_f=make_some_baz):
    ...
    
def make_some_baz(..., swallow_type='european'):
    ...

t=ArgTree
argtree_partial(foo, bar_f=ArgTree(baz_f=t(swallow_type='african)))
```

instead of

```
def foo(..., bar_args, baz_args):
    make_some_bar(..., **bar_args, **baz_args)
    ...

def make_some_bar(..., **baz_args):
    make_some_baz(..., **baz_args)
    ...
    
def make_some_baz(..., swallow_type='european'):
    ...

foo(baz_args=dict(swallow_type='african'))
```
-->
