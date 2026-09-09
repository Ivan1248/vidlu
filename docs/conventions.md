# Code conventions

**Date:** 2026-09-09

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

Boolean attributes and stored state should read as an affirmative statement (`self.is_running`, `target_is_set`), not a bare adjective or a question. This applies to stored state read elsewhere as a proposition; it doesn't override the established PyTorch/sklearn-style constructor-kwarg convention already used throughout the codebase (`bias=False`, `pretrained=True`, `attention=False`), which stays as-is.

Name an identifier after the quantity it represents, not the symbol used for it in a formula: `learning_rate` over `alpha`, `std_dev` over `sigma`. Exceptions: conventional symbols (`x`, `y`) and short loop indices.

Reserve `eps` / `EPS` for a numerical tolerance that absorbs floating-point error or guards near-zero values. Use `threshold` only for a genuine decision boundary between two meaningful regimes.

### 1.2 Function naming by role

Reuse an established verb prefix rather than inventing a new one:

| Prefix | Role | Example |
|---|---|---|
| `compute_` | Pure algorithm / math | `compute_confusion_matrix` |
| `on_` | Event handler / callback | `on_epoch_end` |
| `to_` / `from_` | Type conversion | `to_numpy`, `from_config` |
| `set_` / `update_` | Mutator / partial state change | `set_lr`, `update_metrics` |
| `get_` | Accessor, possibly with args | `get_class_counts` |
| `run_` / `execute_` | Start a long-running process | `run_training` |
| `is_` / `has_` / `can_` | Boolean predicate (affirmative statement) | `is_finite`, `has_labels` |

A method that doesn't read `self` should be a module-level function instead of an instance method.

### 1.3 Factory arguments – the `_f` suffix

A parameter whose value is a *callable that produces an object* (rather than the object itself) is suffixed with `_f`: `block_f`, `backbone_f`, `norm_f`, `data_loader_f`, `lr_scheduler_f`.

This lets callers customize deeply nested construction with `Partial`, `tree_partial`, and `ArgTree` from `vidlu.utils.func`, without a parallel configuration structure:

```python
from vidlu.utils.func import tree_partial, ArgTree as t

def make_flock(swallow_f=make_swallow, ...): ...

au_make_flock = tree_partial(make_flock, swallow_f=t(type='african'))
```

Use `_f` only for arguments that will be *called* to produce the object. Already-constructed values don't take the suffix.

### 1.4 Common type suffixes

Reuse established suffixes before inventing new ones:

| Suffix | Role |
|---|---|
| `*Step` | Training/evaluation step (subclass of `BaseStep`), called as `step(trainer, batch)` |
| `*Extension` | A `TrainerExtension` that augments a `Trainer` |
| `*Config` | Configuration data class (e.g. `TrainerConfig`) |
| `*Maker` | Decoupled constructor requiring late binding (e.g. `OptimizerMaker`) |
| `*Mixin` | Behavior mixin (e.g. `InvertibleModuleMixin`) |
| `*Dataset` | Subclass of `vidlu.data.Dataset` |

### 1.5 Module import aliases

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

## 5. Documentation and prose

**Scope.** [§ 5.1 Writing style (all prose)](#51-writing-style-all-prose) governs *every* piece of prose in this repository: docstrings, inline comments, markdown files under `docs/` and root docs, and commit messages.

### 5.1 Writing style (all prose)

**Punctuation and casing.**

- Always use the spaced en-dash (` – `) – never use the em-dash (`—` / ` — `), and never a double hyphen (`--`) standing in for either.
- Avoid semicolons. Use a comma, a full stop, or an en-dash, whichever the sense calls for.
- Use sentence case for all titles and headings (for example, "How it works", "Training configurations"), capitalising only the first word and proper nouns.

**Content and phrasing.**

- Convey meaning as efficiently as possible: give the essential information, and avoid verbosity, conversational filler, restating what the code plainly says, and redundancy with respect to other documentation.
- Avoid loose abbreviations in prose: write "implementation" rather than "impl", "configuration" rather than "config" when used as a noun in documentation prose.
- Use ISO 24495-1:2023 Plain language or ASD-STE100 Simplified Technical English (STE) where it does not detract from meaning. Consider also the Google developer documentation style guide and the Diátaxis framework for structuring information.
- **Single source of truth applies to comments too.** Document a rule once, at the type or function that owns it, and reference it elsewhere rather than restating it.
- At a call site, keep only what is local to that site – why *this* caller does what it does – and leave the general rule to the function or class being called.
- Prefer plain, accurate language over metaphor and jargon. Use verbs that describe the actual operation (e.g. a tensor is sliced, a batch is collated, an event fires).
- **Name a thing with a noun.** For an operation referred to *as a thing*, use the noun or gerund English already provides, not the bare verb stem: "forward pass" (not "at forward"), "quantization" (not "the quantize"), "encoding" (not "the encode"). Established zero-derived nouns (a build, a call, a step) are exempt.
- Prefer a simple negation to an absolute or temporal adverb: "does not crop", not "never crops". Use temporal adverbs ("always", "never") only for temporal invariants the code enforces.
- **Describe current behaviour only.** After a rename or behaviour change, update affected comments in the same edit, leaving no stale names or accounts of past behaviour. Do not narrate what an earlier buggy version did or why a past change occurred in runtime code comments; record historical context in commit messages or development notes instead.

Where this section is silent, follow the [Google developer documentation style guide](https://developers.google.com/style).

### 5.2 Code documentation (docstrings and comments)

- Google-style docstrings (`"""..."""` with `Args:` / `Returns:` / `Raises:`) for every public class and for functions whose behaviour isn't obvious from the signature. Skip self-documenting methods.
- Document non-obvious parameter contracts: tensor shape (e.g. `(N, C, H, W)`), value range (e.g. `[0, 1]`), or convention.
- Prefer a more descriptive identifier over an inline comment. Add a comment only for rationale a name cannot carry, and explain *why*, not *what*.
- Function docstrings should be descriptive rather than imperative statements (e.g. "Computes the confusion matrix across all batches.", not "Compute the confusion matrix").
- Use two spaces before an inline comment: `x = 1.0  # learning rate`.
- Section banner comments can divide long files:
  ```python
  # Section name ####################################################################################
  ```

### 5.3 Commit messages

Write the subject in the **imperative mood**, as [git-commit](https://git-scm.com/docs/git-commit#_discussion) asks, completing *"if applied, this commit will..."* (`Add`, `Move`, `Fix`, `Upgrade`, `Split`, `Remove`, `Refactor`), not *"if the commit is applied, the code will..."*.

**The verb names what the commit does to the codebase, not what the software does afterwards.**
- One line, capitalised, no trailing period, and no type prefix (this repository does not use `feat:` / `fix:` tags).
- A full descriptive clause rather than a terse label: `Replace eager attention fallback with explicit SDPA preference in VLM loader`, not `Attention fixes`.
- Add a body separated by a blank line for non-trivial changes, explaining what changed and why. Each paragraph is written on one line without hard-wrapping. 
