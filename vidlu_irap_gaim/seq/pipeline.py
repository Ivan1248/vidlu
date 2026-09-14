"""Single-command sequential-enhancement pipeline.

Runs the whole sequential-enhancement stage (feature extraction + caching +
per-attribute LSTM training) from one command, on top of a trained base
classifier experiment:

  python scripts/run.py test <data> <input_adapter> <model> <trainer> -e _ -r \\
    -m "irap_gaim:train_seq_enh,e"

Features are extracted once per (base experiment, checkpoint) into a cache
directory and reused on reruns. Each attribute's LSTM is trained as a regular
vidlu experiment (checkpointing, metrics), independently reproducible with the
printed `run.py train` command.
"""

import hashlib
import json
from pathlib import Path

from irap_data import load_attribute_metadata

from vidlu.metrics import mean_over_defined_attributes

from ..metrics import IRAP_ATTRIBUTE_METRIC_NAMES, IRAP_MAIN_METRIC
from .dataset import DEFAULT_CONTEXT_OFFSETS, resolve_attribute_index
from .feats import extract_features, pack_features


def make_feat_cache_dir(feat_cache_root, experiment_name: str, checkpoint_name: str,
                        feat_dtype: str) -> Path:
    """Resolves and validates the feature cache directory for a base experiment and checkpoint pair.

    Args:
        feat_cache_root: Base cache directory path.
        experiment_name: Identifier string of the base experiment.
        checkpoint_name: Checkpoint name string.
        feat_dtype: Data type string for extracted features ('float16' or 'float32').

    Returns:
        Path to the cache directory containing features and provenance metadata.

    Raises:
        RuntimeError: If existing provenance in the directory does not match arguments.
    """
    key = hashlib.sha1(f"{experiment_name}\n{checkpoint_name}".encode()).hexdigest()[:12]
    cache_dir = Path(feat_cache_root) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Which outputs the directory holds is not recorded here: `extract_features` checks for
    # each segment's files individually, so a cache written before the logits were exported
    # completes itself on the next run instead of needing to be declared stale.
    provenance = dict(experiment=str(experiment_name), checkpoint=str(checkpoint_name),
                      feat_dtype=str(feat_dtype))
    provenance_path = cache_dir / "provenance.json"
    if provenance_path.exists():
        with provenance_path.open() as f:
            recorded = json.load(f)
        differing = {k: (recorded.get(k), v) for k, v in provenance.items()
                     if recorded.get(k) != v}
        if differing:
            raise RuntimeError(
                f"The feature cache {cache_dir} was created with "
                + ", ".join(f"{k}={old!r}, not {new!r}" for k, (old, new) in differing.items())
                + ". Mixing them in one directory would corrupt the packed features. Delete the"
                  " directory to extract again, or pass a different `feat_cache_root`.")
    else:
        with provenance_path.open("w") as f:
            json.dump(provenance, f, indent=2)
    return cache_dir


def get_restored_checkpoint_name(cpman) -> str:
    """Retrieve the name of the checkpoint restored by the checkpoint manager.

    Args:
        cpman: CheckpointManager instance.

    Returns:
        Name of the loaded checkpoint directory.

    Raises:
        RuntimeError: If experiment was not restored from a checkpoint.
    """
    if cpman.resuming_required or cpman.last_index < 0:
        raise RuntimeError(
            "The base experiment was not restored from a checkpoint. Run with `-r` (last"
            " checkpoint) or `-r best`, and the same `-e` suffix as the training run, so that"
            " features are extracted with trained weights.")
    for name in cpman.saved:
        if int(name.split("_")[0]) == cpman.last_index:
            return name
    raise RuntimeError(
        f"No checkpoint with index {cpman.last_index} found among {cpman.saved} in"
        f" {cpman.experiment_dir}.")


# The encoder each input type is read with, as a factory-string fragment. `labels` is a
# predicted class (embedded), `logits` and `feats` are vectors passed through.
INPUT_TYPE_TO_ENCODER_STR = {
    "feats": "irap_gaim.IdentityEncoder(input_dim=data.train.info.feat_dim)",
    "logits": "irap_gaim.IdentityEncoder(input_dim=data.train.info.class_count)",
    "labels": "irap_gaim.LabelEmbeddingEncoder(num_embeddings=data.train.info.class_count)",
}


def make_lstm_experiment_args(*, feat_dir, logit_dir, attribute, metadata_dir, context_offsets,
                              input_types, trainer_str: str, metrics_str: str,
                              main_metrics_str: str, resume, tracker, device):
    """Constructs experiment factory arguments for a single-attribute LSTM training run.

    Args:
        feat_dir: Directory containing extracted features.
        logit_dir: Directory containing the base model's extracted logits.
        attribute: Attribute index or name.
        metadata_dir: Path to metadata directory.
        context_offsets: Offsets of the smoothing window around the classified segment.
        input_types: Base-model outputs the sequence carries ('feats', 'logits', 'labels').
        trainer_str: Trainer factory string.
        metrics_str: Metric factory string.
        main_metrics_str: Primary metric string.
        resume: Checkpoint resume mode.
        tracker: Experiment tracker specification.
        device: Target execution device.

    Returns:
        TrainingExperimentFactoryArgs instance.
    """
    from vidlu.experiments import TrainingExperimentFactoryArgs

    offsets_str = str(tuple(context_offsets)).replace(" ", "")
    input_types_str = str(tuple(input_types)).replace(" ", "")
    data_str = (f"irap_gaim.make_seq_enh_data(feat_dir='{Path(feat_dir).as_posix()}',"
                f"logit_dir='{Path(logit_dir).as_posix()}',"
                f"attribute={attribute!r},metadata_dir='{Path(metadata_dir).as_posix()}',"
                f"context_offsets={offsets_str},input_types={input_types_str})")
    encoders_str = ",".join(f"{name}={INPUT_TYPE_TO_ENCODER_STR[name]}" for name in input_types)
    # `middle_index` comes from the dataset rather than defaulting to the sequence
    # midpoint, so the local-context output is read at the classified segment whatever
    # window `context_offsets` describes.
    model_str = (f"irap_gaim.GeneralLSTMModel,n_classes=data.train.info.class_count,"
                 f"middle_index=data.train.info.target_index,"
                 f"input_encoders=dict({encoders_str})")
    return TrainingExperimentFactoryArgs(
        data=data_str, input_adapter="id", model=model_str, trainer=trainer_str,
        metrics=metrics_str, main_metrics=main_metrics_str, params=None, attach="[]",
        # `alias=module`, not Python's `module as alias`: `parse_aliased_imports_expression`
        # strips spaces and splits on '=', so the latter reaches importlib as one name.
        imports="irap_gaim=vidlu_irap_gaim",
        pre="", experiment_suffix=str(attribute), quick_eval_count=100, resume=resume,
        tracker=tracker, device=device, verbosity=1, deterministic=False,
        factory_version=2, distributed=False)


def _train_lstm_experiment(args, dirs) -> dict:
    """Builds and trains one LSTM experiment; returns its final val/test metrics."""
    from vidlu.experiments import (TrainingExperiment, get_experiment_command,
                                   load_checkpoint_for_resume_mode)

    print("\nLSTM experiment command (standalone-reproducible):\n"
          + get_experiment_command(args))
    lstm_exp = TrainingExperiment.from_args(args, dirs=dirs)
    exit_code = 1
    try:
        training_datasets = {k: v for k, v in lstm_exp.data.items() if k.startswith("train")}
        lstm_exp.trainer.train(*training_datasets.values(), restart=False)

        # Training ends on whichever epoch came last, which need not be a checkpointed one, so
        # the reported numbers are taken from the checkpoint that `resume` designates — the same
        # weights the printed command would restore.
        checkpoint_state, _, index = load_checkpoint_for_resume_mode(lstm_exp.cpman, args.resume)
        lstm_exp.trainer.load_state_dict(checkpoint_state)
        print(f"Evaluating checkpoint {index}"
              f" ({'best' if args.resume == 'best' else 'last'}).")

        split_to_metrics = {}
        for name, ds in lstm_exp.data.items():
            if name.startswith(("val", "test")) and len(ds) > 0:
                eval_state = lstm_exp.trainer.eval(ds, split_name=name)
                split_to_metrics[name] = dict(eval_state.metrics)
        lstm_exp.cpman.remove_old_checkpoints()
        exit_code = 0
    finally:
        if lstm_exp.tracker is not None:
            # Nonzero on failure: a hardcoded 0 here marks a crashed run clean in the tracker.
            lstm_exp.tracker.finish(exit_code=exit_code)
    return split_to_metrics


# The iRAP metrics for the single attribute the LSTM predicts, on top of the `Classification`
# problem defaults (loss and accuracy). Accuracy alone is misleading here: iRAP attributes are
# strongly imbalanced, so a class-frequency predictor already scores high accuracy.
DEFAULT_LSTM_METRICS = "irap_gaim.get_irap_attribute_metrics(data.train.info.class_count)"


def get_evaluated_attribute_indices(base_info) -> list[int]:
    """Retrieve indices of attributes evaluated by the base multi-attribute experiment.

    Args:
        base_info: Dataset `info` mapping containing `attr_to_num_labeled` and `attr_to_value_to_class_idx`.

    Returns:
        List of integer attribute indices with labeled training examples.
    """
    from irap_data.attrs import (get_attrs_to_include, filter_labeled_attrs,
                                 map_attr_names_to_indices)

    names = filter_labeled_attrs(get_attrs_to_include(), base_info["attr_to_num_labeled"])
    return map_attr_names_to_indices(names, list(base_info["attr_to_value_to_class_idx"].keys()))


def summarize_multi_attribute(attribute_to_split_metrics: dict,
                              evaluated_attributes) -> tuple[dict, list[int]]:
    """Computes attribute-averaged multi-attribute metrics across individual attribute models.

    Args:
        attribute_to_split_metrics: Mapping of attribute index to split metric dictionaries.
        evaluated_attributes: Sequence of attribute indices to aggregate.

    Returns:
        Tuple of `(split_to_metrics, missing_attributes)` where `split_to_metrics` maps
        split names to aggregated metric values.

    Raises:
        RuntimeError: If evaluated attributes have inconsistent metric sets.
    """
    missing = [a for a in evaluated_attributes if a not in attribute_to_split_metrics]
    split_to_values, split_to_metric_names = {}, {}
    for attribute in evaluated_attributes:
        for split_name, metric_values in attribute_to_split_metrics.get(attribute, {}).items():
            names = tuple(n for n in IRAP_ATTRIBUTE_METRIC_NAMES if n in metric_values)
            expected = split_to_metric_names.setdefault(split_name, names)
            if names != expected:
                raise RuntimeError(
                    f"On split {split_name!r}, attribute {attribute} has metrics {names} while"
                    f" the attributes before it have {expected}. Averaging them would give each"
                    f" metric a different denominator. Every attribute must be evaluated with"
                    f" the same metrics (`get_irap_attribute_metrics`).")
            metric_to_values = split_to_values.setdefault(split_name, {})
            for name in names:
                metric_to_values.setdefault(name, []).append(metric_values[name])
    # Every metric has one value per attribute (checked above), so any of them counts them.
    # An attribute with an undefined value is dropped from the average, as in
    # `MultiAttributeClassificationMetrics`, so that e.g. `MCC` of an attribute whose split
    # has a single ground-truth class does not make the whole `aMCC` NaN.
    split_to_metrics = {
        split_name: {**{f"a{name}": mean_over_defined_attributes(values)
                        for name, values in metric_to_values.items()},
                     "num_attributes": len(next(iter(metric_to_values.values()), []))}
        for split_name, metric_to_values in split_to_values.items()}
    return split_to_metrics, missing


def train_seq_enh(exp, *, attributes=None, context_offsets=DEFAULT_CONTEXT_OFFSETS,
                  input_types=("feats",), feat_cache_root=None,
                  feat_dtype="float16", lstm_trainer="irap_gaim.seq_enh_lstm_trainer",
                  lstm_metrics=DEFAULT_LSTM_METRICS, lstm_main_metric=IRAP_MAIN_METRIC,
                  lstm_resume="?", lstm_tracker=None, dirs=None):
    """Execute complete sequential enhancement pipeline over a trained base classifier.

    Extracts and caches per-segment feature representations for all dataset splits,
    trains independent `GeneralLSTMModel` temporal smoothing models per attribute,
    and computes aggregated multi-attribute metrics.

    Args:
        exp: Restored base `TrainingExperiment` instance.
        attributes: Optional list of attribute indices or names to train (defaults to all labeled).
        context_offsets: Offsets of the smoothing window around the classified segment.
        input_types: Base-model outputs the sequence carries, any of
            ('feats', 'logits', 'labels').
        feat_cache_root: Optional root directory for feature caching.
        feat_dtype: Numerical precision for cached features ('float16' or 'float32').
        lstm_trainer: Trainer factory specification string for LSTM models.
        lstm_metrics: Metric factory specification string for LSTM evaluation.
        lstm_main_metric: Primary metric for LSTM checkpoint selection.
        lstm_resume: Resume mode string for LSTM checkpoints ('?' to resume existing).
        lstm_tracker: Optional experiment tracker identifier (e.g. 'wandb').
        dirs: Directory paths configuration namespace.

    Returns:
        Dictionary containing `per_attribute` and `multi_attribute` evaluation summaries.
    """
    if dirs is None:
        import dirs  # scripts/dirs.py; importable when running through scripts/run.py

    # Everything the LSTM experiments need is resolved before the (expensive) feature
    # extraction, so a misconfiguration fails immediately instead of hours later.
    # Read individual info keys only: converting the whole (lazy) info mapping could
    # force evaluation of unrelated expensive entries.
    base_info = next(iter(exp.data.values())).info
    metadata_dir = base_info.get("metadata_dir")
    if metadata_dir is None:
        raise RuntimeError(
            "The base dataset does not expose metadata_dir on info; expected it from"
            " irap_data.IRAPDataset.")
    ordered_attrs, _ = load_attribute_metadata(Path(metadata_dir))
    evaluated = get_evaluated_attribute_indices(base_info)
    if attributes is None:
        # Exactly the set the summary averages over, so every trained attribute contributes and
        # `not_trained` can only mean a failure. It is also what the original implementation
        # trained: the base run's attributes intersected with the canonical include list.
        attributes = evaluated
        num_left_out = len(base_info["class_counts"]) - len(attributes)
        if num_left_out > 0:
            print(f"Training the {len(attributes)} attribute(s) that get_irap_metrics evaluates;"
                  f" leaving out {num_left_out} it does not (unlabeled in this release, or"
                  f" outside the canonical subset). Pass `attributes` to train those too.")
    else:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        # Indices throughout, so that the results can be matched with the evaluated set.
        attributes = [resolve_attribute_index(a, ordered_attrs) for a in attributes]
    device = str(next(exp.trainer.model.parameters()).device)

    checkpoint_name = get_restored_checkpoint_name(exp.cpman)
    if feat_cache_root is None:
        feat_cache_root = Path(dirs.cache) / "seq_enh_feats"
    cache_dir = make_feat_cache_dir(feat_cache_root, exp.cpman.experiment_name, checkpoint_name,
                                    feat_dtype)
    # One flat directory per output kind, holding every split: segment IDs are globally
    # unique, so a split is a selection of IDs rather than a directory.
    feat_dir = cache_dir / "feats"
    # The base model's logits, from the same forward pass: they are what the `logits` and
    # `labels` sequence inputs read, and re-running the model to get them would cost a
    # second pass over the dataset.
    logit_dir = cache_dir / "logits"

    print(f"Sequential enhancement: output cache at {cache_dir}")
    for name, dataset in exp.data.items():
        stats = extract_features(exp.trainer, dataset, feat_dir, logit_dir=logit_dir,
                                 dtype=feat_dtype, desc=f"Extracting outputs ({name})")
        print(f"  {name}: {stats['saved']} segments extracted, {stats['skipped']} cached.")
    # Packed once here rather than lazily per attribute, so the cost is paid once and visibly.
    pack_features(feat_dir)
    pack_features(logit_dir)

    attribute_to_metrics = {}
    for attribute in attributes:
        print(f"\n=== Sequential enhancement for attribute"
              f" {ordered_attrs[attribute]!r} (index {attribute}) ===")
        args = make_lstm_experiment_args(
            feat_dir=feat_dir, logit_dir=logit_dir, attribute=attribute,
            metadata_dir=metadata_dir, context_offsets=context_offsets,
            input_types=input_types, trainer_str=lstm_trainer,
            metrics_str=lstm_metrics, main_metrics_str=lstm_main_metric, resume=lstm_resume,
            tracker=lstm_tracker, device=device)
        attribute_to_metrics[attribute] = _train_lstm_experiment(args, dirs)

    multi_attribute, not_trained = summarize_multi_attribute(attribute_to_metrics, evaluated)

    print("\nSequential enhancement summary:")
    for attribute, split_to_metrics in attribute_to_metrics.items():
        print(f"  {ordered_attrs[attribute]} (index {attribute}):")
        for split_name, metric_values in split_to_metrics.items():
            print(f"    {split_name}: "
                  + ", ".join(f"{k}={v}" for k, v in metric_values.items()))
    print(f"\nMulti-attribute metrics, averaged over the {len(evaluated)} attributes evaluated"
          f" by get_irap_metrics (comparable to the base run):")
    if not_trained:
        print("  WARNING: no results for "
              + ", ".join(f"{ordered_attrs[a]} (index {a})" for a in not_trained)
              + "; they are left out of the averages.")
    for split_name, metric_values in multi_attribute.items():
        print(f"  {split_name}: "
              + ", ".join(f"{k}={v}" for k, v in metric_values.items()))
    return dict(per_attribute=attribute_to_metrics, multi_attribute=multi_attribute)
