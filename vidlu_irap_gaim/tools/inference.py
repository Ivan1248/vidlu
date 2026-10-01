"""
Inference-time visualization and structured prediction export.

This module is intended to be used via:
  python scripts/run.py test ... -m irap_gaim.tools.inference

Important: It reuses the trainer's evaluation loop and its configured `eval_step`.
Predictions are collected by hooking into `trainer.evaluation.iter_completed`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
import typing as T

import torch
import torch.nn.functional as F

from irap_data import IGNORE_LABEL_INDEX
from vidlu.experiments import TrainingExperiment
from vidlu.utils.misc import RemovableHandle
from vidlu.utils.collections import NameDict

from vidlu_irap_gaim.tools.vis_utils import (
    PredictionRow,
    make_visualization_image,
    render_prediction_panel_rich,
)
from vidlu_irap_gaim.vlm.response_parser import build_idx_to_value


@dataclass(kw_only=True)
class _PredictOnlyStep:
    """Prediction-only step that works without labels (no loss/metrics dependency)."""

    amp: bool = False

    def __call__(self, trainer, batch):
        import contextlib as ctx
        from vidlu.training.steps import _unify_sup_batch, untag

        trainer.model.eval()
        batch = _unify_sup_batch(batch)
        # Expect a record-like batch with 'rgb'
        x = batch["rgb"] if hasattr(type(batch), "items") else batch[0]

        amp_ctx = torch.cuda.amp.autocast() if self.amp else ctx.nullcontext()
        with amp_ctx, torch.no_grad():
            out = trainer.model(untag(x))
        return NameDict(x=x, target=None, out=out, loss=float("nan"))


@dataclass(kw_only=True)
class _MultiScalePredictOnlyStep:
    """Multi-scale prediction-only step (mirrors MultiScaleSupervisedStep but skips loss)."""

    scales: tuple[float, ...] = (1.0, 0.75, 1 / 0.75)
    amp: bool = False

    def __call__(self, trainer, batch):
        import contextlib as ctx
        from vidlu.training.steps import _unify_sup_batch, untag
        from vidlu_irap_gaim.models.multiscale import MultiScaleSequenceInference

        trainer.model.eval()
        batch = _unify_sup_batch(batch)
        x = batch["rgb"] if hasattr(type(batch), "items") else batch[0]

        # Lazy-create wrapper
        if not hasattr(self, "_ms_model") or self._ms_model is None:
            self._ms_model = MultiScaleSequenceInference(trainer.model, scales=self.scales)

        amp_ctx = torch.cuda.amp.autocast() if self.amp else ctx.nullcontext()
        with amp_ctx, torch.no_grad():
            probs = self._ms_model(untag(x))  # tuple of (B, K_i) probabilities
        return NameDict(x=x, target=None, out=probs, loss=float("nan"))


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _dataset_from_experiment_data(exp_data, split: str):
    # exp.data is a Namespace-like object with .items() and attribute access
    if hasattr(exp_data, split):
        return getattr(exp_data, split)
    if hasattr(exp_data, "get"):
        ds = exp_data.get(split)
        if ds is not None:
            return ds
    if hasattr(exp_data, "items"):
        for name, ds in exp_data.items():
            if name == split:
                return ds
    # fallback to conventional names (Trainer.eval() also has its own fallback)
    if hasattr(exp_data, "test"):
        return getattr(exp_data, "test")
    raise KeyError(f"Dataset split {split!r} not found in experiment.data keys.")


def _extract_rgb_sequence(x) -> torch.Tensor:
    """
    Extract an RGB sequence tensor shaped (B, S, 3, H, W) from the eval_step result.x.

    Supported x shapes/types:
      - Tensor (B,S,3,H,W) or (B,3,H,W)
      - Mapping/Record with key 'rgb' containing a tensor of those shapes
    """
    if isinstance(x, torch.Tensor):
        rgb = x
    elif hasattr(type(x), "items") and "rgb" in x:
        rgb = x["rgb"]
    else:
        raise TypeError(f"Unsupported result.x type for visualization: {type(x)}")

    if not isinstance(rgb, torch.Tensor):
        raise TypeError(f"Expected rgb tensor, got {type(rgb)}")

    if rgb.ndim == 4:
        # (B, 3, H, W) -> (B, 1, 3, H, W)
        rgb = rgb.unsqueeze(1)
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError(f"Expected rgb shaped (B,S,3,H,W), got {tuple(rgb.shape)}")
    return rgb


def _extract_segment_ids(batch) -> list[str]:
    if hasattr(type(batch), "items") and "segment_id" in batch.keys():
        seg = batch["segment_id"]
    else:
        raise KeyError("Batch does not contain 'segment_id' (required for naming outputs).")

    # seg may be list[str], tuple[str], numpy array, or a tensor (unlikely)
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().tolist()
    return [str(s) for s in seg]


def _extract_probs(out: T.Any) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract per-attribute probabilities and argmax predictions from eval_step output.

    - SupervisedStep returns tuple of logits (B, K_i)
    - MultiScaleSupervisedStep returns tuple of probabilities (B, K_i)
    """
    if not isinstance(out, (tuple, list)):
        raise TypeError(f"Expected eval_step output to be tuple/list of tensors, got {type(out)}")

    probs: list[torch.Tensor] = []
    pred_idx: list[torch.Tensor] = []
    pred_prob: list[torch.Tensor] = []

    for o in out:
        if not isinstance(o, torch.Tensor):
            raise TypeError(f"Expected tensor in out tuple, got {type(o)}")
        if o.ndim != 2:
            raise ValueError(f"Expected per-attribute tensor shaped (B,K), got {tuple(o.shape)}")

        # Heuristic: if it already looks like probabilities, don't softmax again.
        # We require non-negative and sums ~ 1.
        with torch.no_grad():
            sum_mean = float(o.sum(dim=-1).mean().detach().cpu().item())
            min_val = float(o.min().detach().cpu().item())
        p = o if (min_val >= -1e-6 and 0.90 <= sum_mean <= 1.10) else F.softmax(o, dim=-1)

        probs.append(p)
        pred_idx.append(p.argmax(dim=-1))
        pred_prob.append(p.max(dim=-1).values)

    return probs, pred_idx, pred_prob


@dataclass(kw_only=True)
class InferenceVisualizationCollector:
    dataset: T.Any
    output_dir: Path
    limit: int | None = None
    save_images: bool = True
    out_size: tuple[int, int] = (1920, 1080)
    text_area_ratio: float = 0.35
    gap: int = 0  # No gap between image and text
    attrs_to_include: tuple[str, ...] | None = None

    def __post_init__(self):
        self.output_dir = _as_path(self.output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        attr_to_value_to_class_idx = self.dataset.info.attr_to_value_to_class_idx
        self._idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)
        self._attr_to_idx = {a: i for i, a in enumerate(attr_to_value_to_class_idx)}

        self.attribute_names = list(attr_to_value_to_class_idx)
        if self.attrs_to_include is not None:
            include = set(self.attrs_to_include)
            self.attribute_names = [a for a in self.attribute_names if a in include]

        self.predictions: dict[str, dict[str, dict[str, T.Any]]] = {}
        self.num_written = 0

        # Bound when registered
        self._evaluation_loop = None

    def bind_evaluation_loop(self, evaluation_loop):
        self._evaluation_loop = evaluation_loop

    def on_iter_completed(self, state) -> None:
        if self.limit is not None and self.num_written >= self.limit:
            if self._evaluation_loop is not None:
                self._evaluation_loop.terminate()
            return

        batch = state.batch
        result = state.result

        seg_ids = _extract_segment_ids(batch)
        rgb_b = _extract_rgb_sequence(result.x)
        probs_list, pred_idx_list, pred_prob_list = _extract_probs(result.out)
        target = getattr(result, "target", None)

        for i, sid in enumerate(seg_ids):
            if self.limit is not None and self.num_written >= self.limit:
                break

            rgb_seq = rgb_b[i].detach().cpu()

            has_target = isinstance(target, torch.Tensor) and target.ndim == 2

            prediction_rows: list[PredictionRow] = []
            # Also build a structured record for JSON
            pred_rec: dict[str, dict[str, T.Any]] = {}

            for attr in self.attribute_names:
                aidx = self._attr_to_idx[attr]
                pred_idx = int(pred_idx_list[aidx][i].detach().cpu().item())
                pred_prob = float(pred_prob_list[aidx][i].detach().cpu().item())
                pred_value = self._idx_to_value[attr][pred_idx]

                gt_value, gt_idx, gt_prob = None, None, None
                if has_target and (label := int(target[i, aidx])) != IGNORE_LABEL_INDEX:
                    gt_idx = label
                    gt_value = self._idx_to_value[attr][gt_idx]
                    gt_prob = float(probs_list[aidx][i, gt_idx].detach().cpu().item())

                prediction_rows.append(
                    PredictionRow(
                        attr=attr,
                        pred_value=pred_value,
                        pred_idx=pred_idx,
                        pred_prob=pred_prob,
                        gt_value=gt_value,
                        gt_idx=gt_idx,
                        gt_prob=gt_prob,
                    )
                )

                pred_rec[attr] = {
                    "pred_idx": pred_idx,
                    "pred_value": pred_value,
                    "prob": pred_prob,
                }
                if gt_idx is not None:
                    pred_rec[attr].update({"gt_idx": gt_idx, "gt_value": gt_value, "gt_prob": gt_prob})

            self.predictions[sid] = pred_rec

            if self.save_images:
                out_img = make_visualization_image(
                    rgb_seq,
                    lambda w, h: render_prediction_panel_rich(prediction_rows, width=w, height=h),
                    out_size=self.out_size, text_area_ratio=self.text_area_ratio, gap=self.gap)
                out_img.save(self.images_dir / f"{sid}_pred.png")

            self.num_written += 1

        if self.limit is not None and self.num_written >= self.limit and self._evaluation_loop is not None:
            self._evaluation_loop.terminate()

    def finalize(
        self,
        *,
        save_json: bool = True,
        metrics: dict[str, T.Any] | None = None,
        split: str | None = None,
    ) -> dict[str, T.Any]:
        summary = {
            "num_written": int(self.num_written),
            "output_dir": str(self.output_dir),
            "split": split,
            "out_size": [int(self.out_size[0]), int(self.out_size[1])],
            "text_area_ratio": float(self.text_area_ratio),
            "save_images": bool(self.save_images),
            "limit": self.limit,
            "metrics": metrics,
        }
        if save_json:
            (self.output_dir / "predictions.json").write_text(
                json.dumps(self.predictions, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return summary


def run(
    experiment: TrainingExperiment,
    *,
    split: str = "test",
    dataset: T.Any = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    save_images: bool = True,
    save_json: bool = True,
    out_width: int = 1920,
    out_height: int = 1080,
    text_area_ratio: float = 0.35,
    attrs_to_include: tuple[str, ...] | None = None,
) -> dict[str, T.Any]:
    """
    Entry point for `scripts/run.py test ... -m irap_gaim.tools.inference`.

    Runs the trainer's normal evaluation (`trainer.eval`) and collects per-sample predictions
    via `trainer.evaluation.iter_completed`.

    Args:
        experiment: The training experiment (provides trainer, model, etc.).
        split: Which split to use from experiment.data (ignored if dataset is provided).
        dataset: Optional custom dataset to run inference on. If provided, `split` is ignored.
            The dataset must have `info.attr_to_value_to_class_idx` for decoding predictions.
            Use `InferenceImageDataset.from_folder()` for custom images.
        output_dir: Where to save results. Defaults to experiment_dir/inference_output.
        limit: Maximum number of samples to process.
        save_images: Whether to save visualization images.
        save_json: Whether to save JSON predictions.
        out_width, out_height: Output image dimensions.
        text_area_ratio: Fraction of width for text panel.
        attrs_to_include: Subset of attributes to visualize.

    Example for custom dataset:
        -m irap_gaim.tools.inference:run,dataset=irap_gaim.InferenceImageDataset.from_folder('/path/to/images',reference_dataset=e.data.test)
    """
    trainer = experiment.trainer
    if dataset is None:
        dataset = _dataset_from_experiment_data(experiment.data, split=split)
    else:
        split = getattr(dataset, "subset", "custom")

    # Detect label-free datasets and switch to prediction-only evaluation.
    #
    # Expectation:
    # - Labeled datasets provide 'target' (Tensor) in records.
    # - Unlabeled datasets omit 'target' entirely.
    #
    # Conclusion:
    # - If unlabeled: disable metrics (often require targets) and swap eval_step to avoid loss.
    has_target = False
    try:
        if len(dataset) > 0:
            ex0 = dataset[0]
            if hasattr(type(ex0), "items"):
                has_target = "target" in ex0.keys()
            elif isinstance(ex0, (tuple, list)) and len(ex0) >= 2:
                has_target = True
    except Exception:
        # If detection fails, keep the default (supervised) behavior and let errors surface.
        has_target = True

    prev_eval_step = None
    prev_metrics = None
    if not has_target:
        prev_eval_step = trainer.eval_step
        prev_metrics = getattr(trainer, "metrics", [])
        # Metrics typically require targets; disable them for unlabeled inference.
        trainer.metrics = []

        # Prefer multi-scale predict-only if the current eval_step looks like multiscale.
        step_name = type(trainer.eval_step).__name__
        is_multiscale = "MultiScale" in step_name
        trainer.eval_step = _MultiScalePredictOnlyStep(amp=getattr(trainer.eval_step, "amp", False)) if is_multiscale else _PredictOnlyStep(amp=getattr(trainer.eval_step, "amp", False))

    out_dir = _as_path(output_dir) if output_dir is not None else _as_path(experiment.cpman.experiment_dir) / "inference_output"
    collector = InferenceVisualizationCollector(
        dataset=dataset,
        output_dir=out_dir,
        limit=limit,
        save_images=save_images,
        out_size=(out_width, out_height),
        text_area_ratio=text_area_ratio,
        attrs_to_include=attrs_to_include,
    )
    collector.bind_evaluation_loop(trainer.evaluation)

    rh: RemovableHandle = trainer.evaluation.iter_completed.add_handler(collector.on_iter_completed)
    try:
        label_note = "" if has_target else " (unlabeled/predict-only)"
        print(f"Evaluating {split} dataset{label_note} and storing results in {out_dir}...")
        trainer.eval(dataset)
    finally:
        rh.remove()
        if prev_eval_step is not None:
            trainer.eval_step = prev_eval_step
        if prev_metrics is not None:
            trainer.metrics = prev_metrics

    metrics = getattr(getattr(trainer, "evaluation", None), "state", None)
    metrics = getattr(metrics, "metrics", None) if metrics is not None else None
    return collector.finalize(save_json=save_json, metrics=metrics, split=split)


