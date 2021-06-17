# ViDLU: Vision Deep Learning Utilities

A deep learning framework for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/).

[![Build Status](https://github.com/Ivan1248/Vidlu/workflows/build/badge.svg)](https://github.com/Ivan1248/Vidlu/actions)
[![codecov](https://codecov.io/gh/Ivan1248/Vidlu/branch/master/graph/badge.svg)](https://codecov.io/gh/Ivan1248/Vidlu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7f89c65e677f490bab26c0e5c7cae116)](https://www.codacy.com/manual/Ivan1248/Vidlu?utm_source=github.com&utm_medium=referral&utm_content=Ivan1248/Vidlu&utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/vidlu/badge/?version=latest)](https://vidlu.readthedocs.io/en/latest/?badge=latest)

This repository contains
1) a machine learning framework mostly based on PyTorch,
1) a set of datasets, models and training configurations (as part of the framework), and
1) a set of scripts that use it

that I am using for research.

## Main scripts

### Running experiments

`scripts/run.py` is a genaral script for running experiments. 

If the `train` procedure is chosen (by running `python run.py train ...`), it forwards its command line arguments and directory paths from `dirs.py` to the `Experiment` constructor, which forwards them to factories from `vidlu.factories` to create a `Trainer` instance. Then `train` runs evaluation and training. If the `--resume` (`--r`) is provided, the `Experiment` instance loads a previously saved training state before continuing training or evaluation.

There is also a `test` procedure that can be used for standard evaluation or for running a custom procedure that can optionally accept the `Experiment` instance as on of its arguments.

`scripts/train_cifar.py` is a specific example where it is easier to tell what is happening.
Running `python train_cifar.py` is equivalent to running

```shell
python run.py train \
    "Cifar10{trainval,test}" "id" \  # data
    "models.ResNetV1,backbone_f=t(depth=18,small_input=True,block_f=t(norm_f=None))" \  # model
    "ct.resnet_cifar,lr_scheduler_f=ConstLR,epoch_count=50,jitter=None"  # training
```

### Directories

`scripts/dirs.py` is a module that determines directory paths needed for running experiments. 

-   `dirs.DATASETS` is a list of paths that can be searched for datasets. One of them is , if defined, in the environment variable `VIDLU_DATASETS`. If found to exist, "&lt;ancestor>/datasets" and "&lt;ancestor>/data/datasets", where "&lt;ancestor>" is any of ancestor directories of `dirs.py`, are included too.
-   `dirs.PRETRAINED` is set to the value of the `VIDLU_PRETRAINED` environment variable if defined or "&lt;ancestor>/data/pretrained_parameters".
-   `dirs.PRETRAINED` is set to the value of the `VIDLU_EXPERIMENTS` environment variable if defined or "&lt;ancestor>/data/experiments".

The following paths are derived: `CACHE = EXPERIMENTS / "cache"` and `SAVED_STATES = EXPERIMENTS / "states"`. They are automatically created by running/importing `dirs.py`.

It might be easiest to create the following directory structure (symbolic links can be useful) so that the directories can be found automatically by `dirs.py`:

```
<ancestor>
├─ .../vidlu/scripts/dirs.py
└─ data
   ├─ datasets
   ├─ experiments  # subdirectories created automatically
   │  ├─ states
   │  └─ cache
   └─ pretrained parameters

```

## The framework

Some of the main packages in the framework are `vidlu.data`, `vidlu.modules`, `vidlu.models`, `vidlu.training`, `vidlu.metrics`, `vidlu.factories`, `vidlu.configs`, `vidlu.utils` and `vidlu.experiment`. 

Most of the code here is rather generic except for concrete datasets in `vidlu.data.datasets`, hyperparameter configurations and other data in `vidlu.configs`, concrete models in `vidlu.models`, some modules in `vidlu.modules.components`, and `vidlu.experiments`, which applies the framework for training and evaluation.

### Data

`vidlu.data` defines types `Record`, `Dataset` and PyTorch `DataLoader`-based types. There are also many concrete datasets in `vidlu.data.datasets`.

`Record` is an ordered key-value mapping that supports lazy evaluation of values. It can be useful when not all fields of dataset examples need to be loaded. 

`Dataset` is the base dataset class. It has a set of useful methods for manipulation and caching (advanced indexing, concatenation, `map`, `filter`, ...). 

`DataLoader` inherits `DataLoader` from PyTorch and changes its `default_collate` so that it supports elements of type `Record`.

### Modules (model components) and models

`vidlu.modules` contains implementations of various modules and functions (`elements`, `components`, `heads`, `losses`) and useful procedures for debugging, extending and manipulating modules. 
\*The modules (inheriting `Module`) support shape inference like in e.g. [MXNet](http://mxnet.incubator.apache.org/) and [MagNet](https://github.com/MagNet-DL/magnet) (an initial run in necessary for initialization). 

`try_get_module_name_from_call_stack` enables getting the name of the current module.

`Seq` is an alternative for `Sequential` which supports splitting, joining, and other things. Many modules are based on it. `deep-split` (accepting a path to some inner module) and `deep_join` can work on composite models that are designed based on `Sequential`.

`with_intermediate_outputs` can be used for extracting intrmediate outputs without changing the module. It uses `register_forward_hook` (and thus requires appropriately designed models).    

For many elementary modules which can be invertible, the `inverse` property returns its inverse module. The inverse is defined either via a `make_inverse` or `inverse_forward`. A `Seq` which consists of only invertible modules (like `Identity`, `Permute`, `FactorReshape`, ...) is automatically invertible. Without a change in the interface, invertible modules also support optional computation and propagation of the logarithm of volume change necessary for normalizing flows.

`vidlu.modules.pert_models` defines parametrized perturbation models that use independent parameters for each input in the mini-batch or mixe data between inputs.

`vidlu.modules.losses` contains loss functions.

Composite modules are desined to be "in-depth" configurable through constructor arguments 
that are factories/constructors for child modules. Their names  usually end with `_f`. Python allows accessing default arguments. If a default argument is a function, its arguments can be modified using e.g. `functools.partial`. `vidlu.utils.func` defines tree data structures and procedures that enable eays modification of deeply nested arguments.

`vidlu.models` contains implementations of some models. Model classes are mostly wrappers around more general modules defined in `vidlu.modules.components` and heads defined in `vidlu.modules.heads`. They also perform initialization of parameters. Some implementad architectures are ResNet-v1, ResNet-v2, Wide ResNet, DenseNet, i-RevNet, SwiftNet<sup>[1](#fn1)</sup>, Ladder-DenseNet<sup>[1](#myfootnote1)</sup>.

<a name="fn1">1</a>: There might be some unintended differences to the original code.

### Training

`vidlu.training` defines procedural machine learning algorithm components.

`Engine` (based on [Ignite](https://github.com/pytorch/ignite)'s `Engine`) is used for running training or evaluation loops (the iteration step is an injected dependency). It raises events before and after the loop and the iteration step.

`Trainer` defines a full machine learning algorithm. It has `train` and `eval` methods. Some of its more important attributes (components) are: `model`, `eval_batch_size` (E), `metrics` (E), `eval_step` (E), `loss` (L),  `batch_size` (L), `jitter` (L), `train_step` (L), `extensions` (L), `epoch_count` (O), `optimizer` (O), `lr_scheduler` (O), `data_loader_f` (D). (E) marks evaluation components, which do not affect training, (L) marks learning components, and (O) marks learning components mostly related to optimization.

`ChecpointManager` is used for storing and loading the state of a trainer (and some other things) in the file system.

`vidlu.training.steps` defines training and evaluation steps. Instances of step classes (inheriting `BaseStep`) have a `__call__` method that accepts a `Trainer` instance and a data batch. Training steps can be stateful and might need to define `state_dict` and `load_state_dict` methods. There are steps of different supervised, adversarial, semi-supervised, normalizing flow, and some hybrid algorithms.

`vidlu.training.robustness` defines 

### Factories

`vidlu.factories` contains the factories that can create model, data, learning, and evaluation components from strings (which can be provided through the command line). Note that it uses Python's `eval` extensively.

`get_prepared_data` accepts a string containing the names of the datasets (with subset names) and code of an arbitrary transformations applied to them (using `Dataset`'s methods, `vidlu.data.utils.dataset_ops`, `vidlu.transforms`, `torchvision.transforms`...). It also requires `datasets_dir` and `cache_dir`, which represent paths to root directories for datasets and cache. It returns a sequence of `Dataset` instances with the transformations applied. The returned `Dataset` instances also convert images and labels to PyTorch `Tensor` instances. They also scale images to range [0..1] and transpose them to the CHW format.

`get_model` (among other arguments) accepts a string containing the name of the model and a list of arguments. The list of arguments is separated by a comma from the model name. The model name should be either (1) a symbol from `vidlu.models`, (2) a symbol reachable through module located in paths listed in the `VIDLU_EXTENSIONS` environment variable, (3) a Pytorch Hub identifier that can be given to `torch.hub.load`. The list of arguments can contain argument trees constructed by nesting calls to the `ArgTree` constructor (`t` is a short alias), or other appropriate `UpdaTree` classes. Some other arguments that `get_model` accepts are `input_adapter_str` (a string defining input pre-processing), `prep_dataset` (prepared dataset used for getting possible problem-specific information and inputs for model initialization) and `device`. 

`get_trainer` accepts a string representing an argument list for the `TrainingConfig` constructor and a model. Keyword arguments can be defined as trees (appropriate instances of `UpdaTree` from `vidlu.utils.func`) that are used to update (without mutation) `TrainingConfig` elements and objects within.

### Training configurations

Training hyperparameter configurations can be defined using the `TrainerConfig` class from `vidlu.configs.training`. The `TrainingConfig` constructor accepts 0 or more `TrainingConfig` instances as positional arguments and keyword arguments that correspond to parameters of the `Trainer` constructor. New configurations can be created by extending updating previously defined ones.

Optimizer configurations can be defined using `OptimizerMaker`, which stores all optimizer information while being decoupled from model parameters. In contrast to optimizers from `torch.optim` and `vidlu.optim` it stores module names instead of parameters. An `optimizer` instance can be created by calling an instance of `OptimizerMaker` with the model as the argument.

`vidlu.configs.training` contains many examples of configurations for things such as classification, semantic segmentation, adversarial training, semi-supervised learning, invertible models, ...

### Experiments

`vidlu.experiment` defines a program for creating and running experiments. It uses `vidlu.factories` to create a `Trainer`, it defines training and evaluation loop actions such as evaluation of performance metrics from `vidlu.metrics`, printing, logging, checkpoint management, user interaction (command execution and training/evaluation step output inspection), and training time estimation. 

### Commonly used utilities

In many places in the code some parameter names end with `_f`. 
This means that the argument is not a final object but a factory (hence `_f`). E.g. `backbone_f()` should produce a `backbone`. This is to allow more flexibility while keeping signatures short. Here the combination of such a design with `ArgTree` and `argtree_partial` allows flexible modification of any set of parameters of nested functions. 

```py

def make_swallow(..., type='european'): ...

def make_flock(..., load=None, swallow_f=make_swallow): ...

def eu(..., fleet_f=make_flock):
    ...
    fleet = fleet_f(...)
    ...

from vidlu.utils.func import ArgTree as t
au = argtree_partial(eu, fleet_f=t(load='coconut', swallow_f=t(type='african')))
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
