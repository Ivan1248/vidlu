# Vidlu configuration system

This document explains how `run.py`, factories, `ArgTree`s, and `TrainerConfig`s work together to enable precise, composable configuration modifications from the command line without requiring a new complete configuration for each experiment.

## Overview

Vidlu uses a hierarchical configuration system that allows:

1. **Composable configurations** – Building complex configurations by combining simpler ones
2. **Deep argument modification** – Modifying nested parameters at any depth without redefining entire structures
3. **CLI-based overrides** – Making precise adjustments via command-line strings evaluated as Python expressions

The system trades simplicity for flexibility: it avoids boilerplate and configuration explosion, but requires understanding several interconnected concepts.

---

## Core components

### 1. `run.py` – the entry point

The script `scripts/run.py` is the main interface for running experiments. It accepts string arguments representing Python expressions that factories interpret to construct objects.

**Basic command structure:**

```sh
python run.py train DATA INPUT_ADAPTER MODEL TRAINER [--params PARAMS] [--metrics METRICS] ...
```

**Example:**

```sh
python run.py train \
    "(Cifar10('trainval'), Cifar10('test'))" "id" \
    "models.ResNetV1,backbone_f=t(depth=18,small_input=True,block_f=t(norm_f=None))" \
    "resnet_cifar,lr_scheduler_f=schedulers.ConstLR,epoch_count=50,jitter=None"
```

The `train` command:
1. Parses string arguments
2. Passes them to factory functions (`get_data`, `get_model`, `get_trainer`)
3. Factories evaluate the strings as Python expressions to create objects
4. Creates a `TrainingExperiment` that runs training and evaluation

### 2. Factories (`vidlu/factories/factories.py`)

Factories convert string expressions into Python objects. They use Python's `eval` to interpret configuration strings within a namespace containing relevant modules and symbols.

**Key factories:**

| Factory | Purpose |
|---------|---------|
| `get_data` | Creates dataset objects from a string describing datasets and subsets |
| `get_model` | Creates a model with an optional `ArgTree` for parameter overrides |
| `get_trainer` | Creates a `Trainer` from a `TrainerConfig` with optional keyword overrides |
| `get_metrics` | Creates metric objects for evaluation |

**Example flow for `get_model`:**

```python
model_str = "models.ResNetV1,backbone_f=t(depth=18,block_f=t(norm_f=None))"
# 1. Parse: model_name = "models.ResNetV1", argtree_arg = "backbone_f=t(depth=18,...)"
# 2. Evaluate argtree_arg in a namespace where t = ArgTree
# 3. Apply the ArgTree to the model factory to create a partial function
# 4. Call the partial function to instantiate the model
```

The factory namespace includes short aliases for common utilities:

```python
_func_short = dict(
    partial=partial,
    t=uf.ArgTree,           # Short for ArgTree
    ft=uf.FuncTree,         # FuncTree
    ot=uf.ObjectUpdatree,   # Object update tree
    ...
)
```

#### Short symbols and the factory namespace (`STANDARD_PRE` + `--pre` / `--imports`)

Names available in factory expressions come from:

1. **The factory's own domain**, injected by each factory as **bare names**:
   - `get_data` → dataset constructors (`Cifar10`, `Cityscapes`, …) plus
     `cache`/`add_pixel_stats`/`extend`/`default_prep`/`standard_prep`;
   - `get_model` → model classes (`ResNetV2`, `SwiftNet`, …);
   - `get_trainer` → trainer configs/steps/attacks from `vidlu.configs.training`
     (`resnet_cifar`, `adversarial`, `madry_Cifar10_attack`,
     `SupervisedTrainMultiStep`, …).
2. **`STANDARD_PRE`** (a statements string `exec`-ed into the shared
   `factory_namespace` by `make_namespace`, `vidlu/factories/factories.py`):
   the DSL/util helpers `partial`, `t`/`ft`/`ot`/`sot`/`it`/`sit` (updatrees),
   `torch`/`np`/`math`; data-expression helpers `rotating_labels`/`chunk`/`folds`/
   `Record`/`class_mapping`/`taxonomies`; and **full-word module handles**:
   `data`, `models`, `modules`, `transforms`, `training`, `configs`, `optim`,
   `components`, `initialization`, `steps`, `attacks`, `jitter`, `schedulers`,
   `losses`. So e.g. `modules.DeconvConv`, `components.PreactBlock`,
   `initialization.kaiming_resnet`, `steps.SemisupVATTrainStep`.
3. installed `vidlu_*` **extensions**, exposed under their de-prefixed name
   (e.g. `irap_gaim.make_bih_data`).
4. the user's `--imports` / `--pre`, which run with/after `STANDARD_PRE` and so
   **extend or override** it.

Examples:

```sh
# Add a symbol not in STANDARD_PRE
--pre "from vidlu.modules.components import SomeBlock"

# Restore a personal short alias if you prefer it
--pre "import vidlu.configs.training as tc"

# Promote an extension name so it can be used unprefixed
--pre "from vidlu_irap_gaim import make_bih_data"
```

### 3. `ArgTree` and Updatrees (`vidlu/utils/func/updatree.py`)

`ArgTree` is a dictionary-like structure that represents parameter modifications to be applied to a callable (factory function). It enables "deep partial application" – modifying arguments of nested factory functions.

**The `_f` convention:**

In Vidlu, parameter names ending with `_f` denote factories (callables that produce objects). For example:
- `backbone_f` – a factory that creates a backbone
- `block_f` – a factory that creates a block (called by `backbone_f`)
- `norm_f` – a factory that creates a normalization layer (called by `block_f`)

This convention enables deep argument trees.

**Example without ArgTree (using nested `partial`):**

```python
from functools import partial as p

backbone_f = p(make_backbone, 
               depth=18, 
               block_f=p(make_block, 
                         norm_f=p(make_norm, eps=1e-5)))
```

**Equivalent with ArgTree (using `t` alias):**

```python
from vidlu.utils.func import ArgTree as t, tree_partial

backbone_f = tree_partial(make_backbone, t(
    depth=18,
    block_f=t(norm_f=t(eps=1e-5))
))
```

**How `ArgTree.apply` works:**

```python
class ArgTree(UpdatreeBase):
    def apply(self, func):
        # Recursively binds arguments to func and its nested _f arguments
        return tree_partial(func, self)
```

When applied to a function, an `ArgTree` binds top-level keys as keyword arguments and recursively applies nested `ArgTree`s to corresponding factory arguments.

### 4. `TrainerConfig` (`vidlu/configs/training/trainer_config.py`)

`TrainerConfig` is a dictionary-like class for composing training configurations. It supports:

1. **Inheritance via positional arguments** – Passing other `TrainerConfig` instances merges their settings
2. **Extension factories** – Special handling for trainer extensions (passed as positional arguments or in `extension_fs`)
3. **Keyword overrides** – Later values override earlier ones

**Definition:**

```python
class TrainerConfig(NameDict):
    def __init__(self, *args, **kwargs):
        # Extension factories are concatenated in order
        ext_args = []
        all_kwargs = {}
        for x in args:
            if isinstance(x, TrainerConfig):
                d = dict(**x)
                ext_args.extend(d.pop('extension_fs', ()))
                all_kwargs.update(d)
            elif issubclass(x, TrainerExtension):
                ext_args.append(x)
        all_kwargs.update(kwargs)
        super().__init__(**all_kwargs, extension_fs=tuple(ext_args) + tuple(kwargs.get('extension_fs', ())))
```

**Composition example:**

```python
# Base configuration
supervised = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.supervised_train_step,
)

# Classification extends supervised
classification = TrainerConfig(
    supervised,  # inherits eval_step and train_step
    loss=losses.nll_loss_l
)

# CIFAR ResNet extends classification
resnet_cifar = TrainerConfig(
    classification,  # inherits from supervised + classification
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=1e-4),
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128,
    jitter=jitter.CifarPadRandCropHFlip(),
)

# Cosine variant overrides only lr_scheduler_f
resnet_cifar_cosine = TrainerConfig(
    resnet_cifar,
    lr_scheduler_f=partial(CosineLR, eta_min=1e-4),  # override
)
```

### 5. Dataset Configuration (`get_data`)

The `get_data` factory evaluates the data string as a Python expression in a namespace containing dataset factories, transforms, and utilities.

**Available in the namespace:**

- **Dataset factories**: `Cifar10`, `Cityscapes`, `CamVid`, `MNIST`, etc. (from `vidlu.data.DatasetFactory`). The subset is the first positional argument, e.g. `Cifar10('trainval')`.
- **Default transforms**: `cache`, `add_seg_class_info`, `add_pixel_stats`, `extend`, `default_prep`, `standard_prep` (for manual / override use; see below)
- **Transform modules**: `transforms` (`vidlu.transforms`; image transforms via `transforms.image`)
- **Dataset operations** (bare, from `vidlu.data.utils.dataset_ops`): `rotating_labels`, `chunk`, `chunks`, `folds`, `remap_classes`, `add_class_mapping`

**Basic syntax (current, factory_version=2):**

The data string is a Python expression that returns either:
- A sequence of datasets `(train_ds, test_ds)` – auto-named as `train`, `test`
- A mapping of named datasets; prefer the `dict(...)` form over `{'...': ...}`
  since all split names (`train`, `train_u`, `val`, `test`, ...) are valid
  identifiers and the keyword form is less noisy.

The expression declares **only which datasets** to use. Pixel statistics are
added to training splits centrally afterwards — see "Default preparation" below —
so data strings no longer need to wrap datasets in `add_pixel_stats(...)`. Example
HDD caching is opt-in (`prep=standard_prep`, or `cache(...)` per dataset).

```sh
# Simple: explicit train/test (returns tuple)
"(Cifar10('trainval'), Cifar10('test'))"

# With constructor arguments
"(Cityscapes('train', downsampling=2), Cityscapes('val', downsampling=2))"

# Multiple test sets (prefer explicit naming via dict)
"dict(train=Cifar10('trainval'), test0=Cifar10('test'), test1=Cifar100('test'))"
```

**Default preparation (`get_prepared_data_for_trainer`):**

After the expression is evaluated and splits are named, each split is prepared by
`get_default_data_prep` (`vidlu/factories/factories.py`):

- splits whose name starts with `train` get pixel statistics added **lazily**
  (`add_pixel_stats_to_info_lazily`) — read only by the `standardize` input
  adapter and only from the training split, so it is free for `id`-adapter runs.
  Splits whose dataset already provides `info.pixel_stats` (e.g. IRAP datasets)
  are skipped;
- example HDD caching is **opt-in** — the default no longer caches. Use
  `standard_prep` (the previous default: pixel stats + caching) or the `cache`
  helper to cache specific datasets.

You can override this stage with a reserved `prep` item in the mapping — a
callable `prep(datasets: dict) -> dict` applied to the whole named-split mapping
(and removed before the splits are prepared). `default_prep` (pixel stats only)
and `standard_prep` (pixel stats + caching) are exposed in the namespace so
overrides can compose with them:

```sh
# Opt into the previous behaviour (cache every split that supports it)
"dict(train=Cifar10('trainval'), test=Cifar10('test'), prep=standard_prep)"

# Cache only the training split (cache() wraps a single dataset)
"dict(train=cache(Cifar10('trainval')), test=Cifar10('test'))"
```

The whole-mapping signature also makes `prep` the right place for any cross-split
logic (e.g. computing a statistic on `train` and writing it into `val`/`test`);
the default does none.

**Advanced: transformations and slicing**

Since data strings are Python expressions, you can apply Dataset methods directly:

```sh
# Subset of training data
"(Cifar10('trainval')[:1000], Cifar10('test'))"

# Using dataset operations (permute, filter, etc.)
"(Cityscapes('train').permute(42)[:500], Cityscapes('val'))"
```

**Examples using `vidlu.data.utils.dataset_ops`**

Some dataset operations from `vidlu.data.utils.dataset_ops` are directly available in the data-expression namespace. This enables concise dataset splitting and re-indexing, typically applied to the training set.

```sh
# K-fold split on training data only
"dict(train=folds(Cifar10('trainval'), 5)[0], val=folds(Cifar10('trainval'), 5)[1], test=Cifar10('test'))"

# Train/val split via slicing (Cifar10 has only 'trainval'/'test', so split trainval)
"dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])"

# Semi-supervised: labeled subset + full unlabeled training set
"dict(train=Cifar10('trainval')[:4000], train_u=Cifar10('trainval'), test=Cifar10('test'))"

# Label re-ordering for uniform-label datasets (semi-supervised setup)
"dict(train=rotating_labels(Cifar10('trainval'))[:4000], train_u=Cifar10('trainval'), test=Cifar10('test'))"
```

**Returning a dict for explicit naming:**

```python
# Expression returning a dict
"dict(train=Cifar10('trainval'), val=Cifar10('test')[:500], test=Cifar10('test')[500:])"
```

**Semi-supervised and multi-dataset scenarios:**

For semi-supervised learning, you typically need `train` (labeled), `train_u` (unlabeled), and `test`:

```sh
# Dict with explicit names
"dict(train=Cityscapes('train')[:100], train_u=Cityscapes('train'), test=Cityscapes('val'))"
```

**Legacy syntax (deprecated / removed from current `vidlu.factories`):**

The old syntax used colons to separate components:

```
"names:dataset_spec:transform_expr"
```

Example from `scripts/commands/commands.sh`:

```sh
# OLD SYNTAX (historical) - not supported by the current `vidlu.factories.get_data`
"train,train_u,test:Cifar10{trainval,test}:(rotating_labels(d[0])[:4000],d[0],d[1])"
```

This means:
- Names: `train`, `train_u`, `test`
- Datasets: `Cifar10{trainval,test}` → `d[0]` = trainval, `d[1]` = test
- Transform: `(rotating_labels(d[0])[:4000], d[0], d[1])` – applies `rotating_labels`, takes first 4000 samples

**Note:** `scripts/run.py` defaults to `factory_version=2` and the active implementation in `vidlu.factories.get_prepared_data_for_trainer` asserts `factory_version > 1`, so the colon-based syntax must be rewritten into Python expressions (typically dicts) to work with the current code.

### 6. How CLI Modifications Work

When you run:

```sh
python run.py train "(Cifar10('trainval'), Cifar10('test'))" "id" \
    "ResNetV1,depth=34" \
    "resnet_cifar,epoch_count=100,optimizer_f=t(lr=0.05)"
```

The trainer string `"resnet_cifar,epoch_count=100,optimizer_f=t(lr=0.05)"` is processed by `get_trainer` (which injects `vidlu.configs.training` names, so `resnet_cifar` resolves bare):

```python
def get_trainer(trainer_str: str, ...):
    # 1. Evaluate the string as an ArgHolder. The namespace includes vars(ct),
    #    so trainer-config names resolve bare. (ct = vidlu.configs.training)
    ah = factory_eval(f"uf.ArgHolder({trainer_str})", {**vars(ct), **namespace, **_func_short, 'uf': uf})
    # ah.args = (ct.resnet_cifar,)
    # ah.kwargs = {'epoch_count': 100, 'optimizer_f': ArgTree(lr=0.05)}
    
    # 2. Create base config from positional args
    config = ct.TrainerConfig(*ah.args)
    
    # 3. Apply keyword overrides via ObjectUpdatree
    updatree = uf.ObjectUpdatree(**ah.kwargs)
    config = updatree.apply(config)
    # This sets config['epoch_count'] = 100
    # and applies ArgTree(lr=0.05) to config['optimizer_f']
    
    # 4. Normalize and create Trainer
    trainer_f = partial(Trainer, **config.normalized())
    return trainer_f(model=model, ...)
```

---

## Complete data flow

```
CLI String Arguments
        │
        ▼
   run.py (argparse)
        │
        ▼
 TrainingExperiment.from_args()
        │
        ├──► get_data(data_str) ──► Dataset objects
        │
        ├──► get_model(model_str) ──► Model instance
        │         │
        │         └──► ArgTree.apply() for deep parameter binding
        │
        └──► get_trainer(trainer_str, model) ──► Trainer instance
                  │
                  ├──► TrainerConfig composition
                  └──► ObjectUpdatree.apply() for overrides
```

---

## Pros and cons

### Advantages

| Aspect | Description |
|--------|-------------|
| **Flexible configurability** | Modify any parameter at any nesting depth without redefining entire configurations |
| **Composition over inheritance** | Build complex configs by combining simpler ones; avoids configuration explosion |
| **Boilerplate minimization** | No need for separate config files for every variant; modifications are inline |
| **Generality** | The same mechanism works for models, trainers, optimizers, data pipelines, etc. |
| **CLI expressiveness** | Full Python expressions available at the command line |
| **Extensibility** | Extensions can add new symbols to the factory namespace |

### Disadvantages

| Aspect | Description |
|--------|-------------|
| **Complexity** | Understanding requires knowledge of `ArgTree`, `TrainerConfig`, factories, and their interactions |
| **Non-standard** | Not a widely-used pattern; new contributors face a learning curve |
| **Debugging difficulty** | Errors in string expressions can be hard to trace; stack traces may be unclear |
| **IDE support** | Limited autocompletion and static analysis for string-based configurations |
| **Security** | Uses `eval` extensively; not suitable for untrusted input |
| **Documentation** | The `_f` convention and deep argument trees require explanation |
| **Implicit behavior** | `TrainerConfig.normalized()` binds parameters to extension factories implicitly |

---

## Usage patterns

### Pattern 1: Quick hyperparameter sweep

```sh
# Vary learning rate without defining new configs
python run.py train ... "resnet_cifar,optimizer_f=t(lr=0.01)"
python run.py train ... "resnet_cifar,optimizer_f=t(lr=0.05)"
python run.py train ... "resnet_cifar,optimizer_f=t(lr=0.1)"
```

### Pattern 2: Model architecture modification

```sh
# Change ResNet depth and disable batch normalization
python run.py train ... \
    "models.ResNetV1,backbone_f=t(depth=50,block_f=t(norm_f=None))"
```

### Pattern 3: Combining semi-supervised and adversarial training

```python
# In vidlu/configs/training/_training.py
semisup_adversarial = TrainerConfig(
    semisup_vat,
    adversarial,  # adds AdversarialTraining extension
    train_step=ts.SemisupAdversarialStep(),
)
```

### Pattern 4: Extending from CLI

```sh
# Add an extension and override parameters
python run.py train ... \
    "swiftnet_cityscapes,adversarial,attack_f=attacks.PGD"
```

---

## Key files reference

| File | Purpose |
|------|---------|
| `scripts/run.py` | Entry point; argument parsing and experiment execution |
| `scripts/commands/commands.sh` | Example commands (some may use deprecated syntax) |
| `vidlu/factories/factories.py` | Factory functions that parse strings into objects |
| `vidlu/utils/func/updatree.py` | `ArgTree`, `FuncTree`, and related update tree classes |
| `vidlu/configs/training/trainer_config.py` | `TrainerConfig` class definition |
| `vidlu/configs/training/_training.py` | Predefined training configurations |
| `vidlu/experiments.py` | `TrainingExperiment` class that orchestrates training |
| `vidlu/data/` | Dataset classes and `DatasetFactory` |

---

## Notes on `scripts/commands/commands.sh`

The CIFAR10 commands in `commands.sh` have been migrated to `factory_version=2`.
Other sections (e.g. some Cityscapes lines) may still use the **deprecated**
`factory_version=1` colon syntax.

**Examples using deprecated colon syntax:**

```sh
# Deprecated (factory_version=1):
"train,train_u,test:Cifar10{trainval,test}:(rotating_labels(d[0])[:4000],d[0],d[1])"
"train,train_u,test:Cityscapes{train,val}:(d[0],d[0],d[1])"
```

**Updated equivalents (factory_version=2):**

Pixel statistics are now added centrally (see "Default preparation") and caching
is opt-in, so neither is written into the data string (add `prep=standard_prep`
to cache):

```sh
# Cifar10 semi-supervised:
"dict(train=rotating_labels(Cifar10('trainval'))[:4000], train_u=Cifar10('trainval'), test=Cifar10('test'))"

# Cityscapes semi-supervised (pixel-stats handled centrally; caching opt-in):
"dict(train=Cityscapes('train'), train_u=Cityscapes('train'), test=Cityscapes('val'))"
```

**Examples using current syntax:**

```sh
# Subset is the first positional argument
"(Cifar10('trainval'), Cifar10('test'))"
"(Cityscapes('train', downsampling=2), Cityscapes('val', downsampling=2))"

# CamVid has no 'trainval' subset (only train/val/test); join train+val for it:
"(CamVid('train').join(CamVid('val')), CamVid('test'))"
```

Note: use the registered class names exactly (`CamVid`, `MNIST`, ...); lowercase
aliases like `camvid`/`mnist` are not in the evaluation namespace.

When updating old commands, replace the colon-based syntax with a Python
expression (a tuple or `dict(...)`) that returns the datasets; the central default
prep adds pixel statistics, and caching is opt-in (`prep=standard_prep`).

---

## Summary

The Vidlu configuration system provides powerful, composable configuration through:

1. **String-based factory arguments** evaluated as Python expressions
2. **`ArgTree`s** for deep parameter modification of nested factories
3. **`TrainerConfig`s** for composing training settings with inheritance-like semantics
4. **`ObjectUpdatree`s** for applying overrides to existing configurations

This approach minimizes configuration boilerplate and enables CLI-based hyperparameter tuning, but requires understanding the underlying mechanisms and conventions.
