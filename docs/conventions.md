# Code conventions

Conventions for keeping the Vidlu codebase consistent. The patterns below already exist in the code; this document makes them explicit.

See [`../AGENTS.md`](../AGENTS.md) for broader coding guidelines that apply across projects.

---

## 1. Naming

### 1.1 General (PEP 8)

- `PascalCase` for classes.
- `snake_case` for functions, methods, variables, modules, and file names.
- `UPPER_SNAKE_CASE` for module-level constants.
- Leading `_` for private module members and attributes.

Names used outside a small local scope should be specific: `compute_confusion_matrix`, not `compute_cm`.

### 1.2 Factory arguments – the `_f` suffix

A parameter whose value is a *callable that produces an object* (rather than the object itself) is suffixed with `_f`: `block_f`, `backbone_f`, `norm_f`, `data_loader_f`, `lr_scheduler_f`.

This lets callers customize deeply nested construction with `Partial`, `tree_partial`, and `ArgTree` from `vidlu.utils.func`, without a parallel configuration structure:

```python
from vidlu.utils.func import tree_partial, ArgTree as t

def make_flock(swallow_f=make_swallow, ...): ...

au_make_flock = tree_partial(make_flock, swallow_f=t(type='african'))
```

Use `_f` only for arguments that will be *called* to produce the object. Already-constructed values don't take the suffix.

### 1.3 Common type suffixes

Reuse established suffixes before inventing new ones:

| Suffix | Role |
|---|---|
| `*Step` | Training/evaluation step (subclass of `BaseStep`), called as `step(trainer, batch)` |
| `*Extension` | A `TrainerExtension` that augments a `Trainer` |
| `*Config` | Configuration data class (e.g. `TrainerConfig`) |
| `*Maker` | Decoupled constructor requiring late binding (e.g. `OptimizerMaker`) |
| `*Mixin` | Behavior mixin (e.g. `InvertibleModuleMixin`) |
| `*Dataset` | Subclass of `vidlu.data.Dataset` |

### 1.4 Module import aliases

Imports of `vidlu` subpackages follow a `v` + submodule-initials pattern. Reuse the established alias rather than inventing a new one:

| Import | Alias |
|---|---|
| `vidlu.modules` | `vm` |
| `vidlu.modules.utils` | `vmu` |
| `vidlu.modules.components` | `vmc` |
| `vidlu.modules.losses` | `vml` |
| `vidlu.training` | `vt` |
| `vidlu.training.steps` | `ts` |
| `vidlu.training.extensions` | `te` |
| `vidlu.data` | `vd` |
| `vidlu.data.utils` | `vdu` |
| `vidlu.utils.func` | `vuf` |
| `vidlu.torch_utils` | `vtu` |
| `vidlu.configs.training` | `vct` |
| `vidlu.utils.distributed` | `vud` |

Also: `import typing as T`, `import numpy as np`, `import dataclasses as dc`.

---

## 2. Package layout

Group by domain subsystem, not by technical layer:

```
vidlu/
├─ data/         Record, Dataset, DataLoader, concrete datasets
├─ modules/      nn.Module subclasses, components, losses, perturbation models
├─ models/       Concrete architectures (thin wrappers over modules.components)
├─ training/     Trainer, EpochLoop, steps, extensions, CheckpointManager
├─ configs/      TrainerConfig and concrete training configurations
├─ transforms/   Input/output transforms
├─ factories/    String → object construction (uses eval)
├─ optim/        Optimizers, LR schedulers
├─ ops/          Elementary tensor operations
└─ utils/        Cross-cutting helpers (func, collections, num, path, …)
```

Place subsystem-specific logic in its subsystem; reserve `utils/` for genuinely cross-cutting helpers.

Use section banner comments for navigating long files:

```python
# Section name ####################################################################################
```

Extensions are loaded from external packages named `vidlu_*` (see `vidlu.extensions`).

---

## 3. Module design

### 3.1 Shape inference

Framework `Module` subclasses initialize on the first forward pass. Defer shape-dependent parameter creation until shapes are known rather than declaring them eagerly.

### 3.2 Invertibility

An invertible module exposes its inverse via the `inverse` property, implemented by defining either `make_inverse` or `inverse_forward`. A `Seq` of invertible modules is automatically invertible.

### 3.3 Composition over parallel configuration

Prefer composing callables with `Partial`, `tree_partial`, and `ArgTree` over introducing a separate configuration structure that mirrors a function's signature.

---

## 4. Physical quantities and units

- Use SI internally. When a non-SI unit is unavoidable, encode it in the name with a snake_case suffix: `duration_s`, `interval_ms`, `angle_deg`.
- Convert non-SI inputs to SI at the boundary.
- Use `YYYY-MM-DD` for dates in filenames, docstrings, and notes.

---

## 5. Documentation

- Google-style docstrings (`"""..."""` with `Args:` / `Returns:`) for every public class and for functions whose behavior isn't obvious from the signature.
- Document non-obvious parameter contracts: tensor shape (e.g. `(N, C, H, W)`), value range (e.g. `[0, 1]`), or convention.
- Prefer a more descriptive name over a comment; add a comment only for rationale a name cannot carry.
