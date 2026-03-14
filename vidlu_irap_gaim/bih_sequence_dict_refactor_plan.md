# Refactor BihSequence to use dict instead of Record

## Goal Description
Update `BihSequence` in `vidlu_irap_gaim` to return standard Python `dict`s instead of `vidlu.data.Record` objects. This decouples the dataset implementation from the `Record` class, potentially simplifying dependencies and usage. We ensures compatibility by updating `make_sequence_color_jitter` to handle `dict` inputs.

## User Review Required
> [!IMPORTANT]
> This change modifies the return type of `BihSequence.get_example` from `Record` to `dict`. Downstream code relying on `Record`-specific features (like dot notation access `x.rgb` instead of `x['rgb']` or `type(x)(x, **kwargs)` constructor) will need updates. I have identified `make_sequence_color_jitter` as one such place.

## Proposed Changes

### vidlu_irap_gaim

#### [MODIFY] [datasets.py](file:///d:/GoogleDrive/Projects/vidlu/vidlu_irap_gaim/datasets.py)
- Change `BihSequence.get_example` to return `dict` instead of `Record`.
- Remove `Record` import if unused or update imports.

#### [MODIFY] [training.py](file:///d:/GoogleDrive/Projects/vidlu/vidlu_irap_gaim/training.py)
- Update `make_sequence_color_jitter._apply` to handle `dict` inputs.
  - If input is `dict`, use `d.copy()` and update key.
  - If input is `Record`, preserve existing behavior.

## Verification Plan

### Automated Tests
- Run `pytest vidlu_irap_gaim/tests/test_semisup.py` to ensure no regressions in existing tests.
- Create and run a verification script `verify_dict_refactor.py` that:
    1. Instantiates `BihSequence`.
    2. Fetches an example and asserts it is a `dict`.
    3. Runs `make_sequence_color_jitter` on the dict and asserts it works.

### Manual Verification
- Run the first training command from `commands.md` for a few steps to ensure the training loop accepts the `dict` data.
  ```bash
  python scripts/run.py train "irap_gaim.make_bih_data()" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics()" --debug --epoch_count 1
  ```
