"""
Inference + visualization utility for IRAP BiH sequences.

This is a vidlu-native port of `libs/irap_gaim-main-m/inference_visualization.py`:
- Uses `irap_data.make_bih_data` (instead of DatasetWrapper/ImageSequenceDataset).
- Uses `vidlu_irap_gaim.models.classification.ImageSequenceClassifier` for "local" mode.
- Loads checkpoints from Vidlu's `CheckpointManager` format (supports `model_state.pth`).

The script writes per-segment images combining:
- a grid of context RGB frames
- a text panel with predicted attribute values
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from irap_data.irap_dataset import RGB_MEAN, RGB_STD
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as a script from repo root without installation (match dataset_viewer.py behavior)
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from irap_data.attrs import get_attrs_to_include  # noqa: E402
from vidlu_irap_gaim.compat.legacy_seq_enh_model import LegacyGeneralLSTMModel  # noqa: E402
from irap_data import make_bih_data, resolve_irap_paths  # noqa: E402
from vidlu_irap_gaim.models.classification import ImageSequenceClassifier  # noqa: E402
from vidlu_irap_gaim.models.encoders.resnet import ResNetEncoder  # noqa: E402
from vidlu_irap_gaim.tools.vis_utils import AttributeMetadataDecoder, save_inference_visualization  # noqa: E402


def _parse_int_list_csv(x: str) -> tuple[int, ...]:
    if x.strip() == "":
        return tuple()
    return tuple(int(s.strip()) for s in x.split(",") if s.strip() != "")


def _standardize_rgb(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Standardize an RGB sequence tensor.

    Args:
        x: (B, S, 3, H, W) in [0, 1]
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)[None, None, :, None, None]
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype)[None, None, :, None, None]
    return (x - mean_t) / std_t


def _load_model_state_dict(model_state_path: str | Path, *, map_location: str | torch.device):
    p = Path(model_state_path)
    state = torch.load(p, map_location=map_location)
    # Vidlu checkpoint manager saves `model_state.pth` as a plain state_dict.
    # Some external checkpoints may wrap it.
    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        return state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint content type {type(state)} at {p}")
    return state


def run_local(args) -> None:
    ds_dir, md_dir = resolve_irap_paths(dataset_dir=args.dataset_dir, metadata_dir=args.metadata_dir)
    datasets = make_bih_data(
        dataset_dir=ds_dir,
        metadata_dir=md_dir,
        context_offsets=args.context_offsets,
        data_types=("rgb",),
        attribute_value_mapping_path=args.attribute_value_mapping_path,
        use_ncontext_filter=not args.no_ncontext_filter,
        seg_to_res_path=args.seg_to_res_path,
    )
    ds = datasets[args.split]
    decoders = AttributeMetadataDecoder.load(
        metadata_dir=md_dir,
        attribute_value_mapping_path=args.attribute_value_mapping_path,
        expected_attribute_names=ds.info.attribute_names,
    )

    attrs_to_include = set(get_attrs_to_include())
    attribute_names = list(ds.info.attribute_names)
    attribute_to_idx = {a: i for i, a in enumerate(attribute_names)}

    # Model: match training defaults (sequence_length = len(context_offsets))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageSequenceClassifier(
        class_counts=tuple(ds.info.class_counts),
        attention=args.attention,
        sequence_length=len(args.context_offsets),
        encoder_f=lambda: ResNetEncoder(pretrained=args.pretrained_backbone),
    ).to(device)

    # Load model weights
    if args.model_state_path is None and args.checkpoint_dir is None:
        raise ValueError("Provide either --model_state_path or --checkpoint_dir.")
    model_state_path = (
        Path(args.model_state_path)
        if args.model_state_path is not None
        else Path(args.checkpoint_dir) / "model_state.pth"
    )
    state_dict = _load_model_state_dict(model_state_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.verbose:
        print(f"Loaded model state: {model_state_path}")
        if missing:
            print(f"Missing keys (count={len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys (count={len(unexpected)}): {unexpected[:10]}")

    # DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    n_done = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"local:{args.split}"):
            rgb = batch["rgb"].to(device)
            seg_ids = batch["segment_id"]

            if args.input_adapter == "standardize":
                rgb_in = _standardize_rgb(rgb, mean=args.mean, std=args.std)
            elif args.input_adapter == "id":
                rgb_in = rgb
            else:
                raise ValueError(f"Unknown input_adapter: {args.input_adapter}")

            outputs = model({"rgb": rgb_in})  # tuple of (B, K_i)

            for i in range(rgb.shape[0]):
                lines: list[str] = []
                for attr in attribute_names:
                    if attr not in attrs_to_include:
                        continue
                    aidx = attribute_to_idx[attr]
                    pred_idx = int(outputs[aidx][i].argmax(dim=-1).item())
                    lines.append(decoders.to_text(attr=attr, class_idx=pred_idx))
                text = "\n".join(lines)
                out_path = save_inference_visualization(
                    rgb_seq=rgb[i].detach().cpu(),
                    text=text,
                    segment_id=str(seg_ids[i]),
                    output_dir=args.output_dir,
                    out_size=(args.out_width, args.out_height),
                    text_area_ratio=args.text_area_ratio,
                )
                n_done += 1
                if args.limit is not None and n_done >= args.limit:
                    if args.verbose:
                        print(f"Reached --limit={args.limit}. Last output: {out_path}")
                    return

    if args.verbose:
        print(f"Saved {n_done} visualizations to {args.output_dir}")


def _load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _build_attribute_to_checkpoint_legacy(config: Mapping, root_dir: str | Path) -> dict[str, Path]:
    root_dir = Path(root_dir)
    model_save_name = config.get("model_save_name", None)
    if model_save_name is None:
        # Some configs omit this; treat root_dir as the model_save_name directory itself.
        model_root = root_dir
    else:
        model_root = root_dir / model_save_name

    out: dict[str, Path] = {}
    for attribute in config["attribute_to_n_classes"]:
        attr_dir = attribute.replace("/", "--").replace(" ", "_")
        out[attribute] = model_root / attr_dir / "best_model_MF1.pt"
    return out


def _load_feat_sequence(feat_dir: str | Path, context_ids: Sequence[str]) -> torch.Tensor:
    feat_dir = Path(feat_dir)
    frames = []
    for sid in context_ids:
        p = feat_dir / f"{sid}.npy"
        arr = np.load(p).reshape(-1).astype(np.float32)
        frames.append(arr)
    return torch.from_numpy(np.stack(frames, axis=0))  # (S, D)


def run_sequential_legacy(args) -> None:
    """
    Sequential mode (legacy checkpoints):
    - Uses legacy per-attribute LSTM smoothing checkpoints (best_model_MF1.pt).
    - Uses exported per-segment feature vectors from `--feat_dir` and a long `--context_offsets`.
    - Still visualizes RGB frames from BiH.
    """
    if args.seq_config_path is None:
        raise ValueError("--seq_config_path is required for mode=sequential_legacy")
    if args.seq_models_root is None:
        raise ValueError("--seq_models_root is required for mode=sequential_legacy")
    if args.feat_dir is None:
        raise ValueError("--feat_dir is required for mode=sequential_legacy")

    config = _load_json(args.seq_config_path)
    if "experiment_config" in config and "attribute_to_lstm_hyperparams" not in config:
        config = config["experiment_config"]

    max_N = int(config.get("max_N", 10))
    N = int(config["evaluated_Ns"][0] if "evaluated_Ns" in config else 10)

    # Dataset for visualization + context ids (must match max_N layout expected by legacy model)
    ds_dir, md_dir = resolve_irap_paths(dataset_dir=args.dataset_dir, metadata_dir=args.metadata_dir)
    datasets = make_bih_data(
        dataset_dir=ds_dir,
        metadata_dir=md_dir,
        context_offsets=args.context_offsets,
        data_types=("rgb",),
        attribute_value_mapping_path=args.attribute_value_mapping_path,
        use_ncontext_filter=not args.no_ncontext_filter,
        seg_to_res_path=args.seg_to_res_path,
    )
    ds = datasets[args.split]

    # Validate legacy context length assumptions: expects full window 2*max_N+1 and 0 offset at index max_N
    if len(args.context_offsets) != 2 * max_N + 1:
        raise ValueError(
            f"Legacy sequential models expect context_offsets length 2*max_N+1.\n"
            f"Got len(context_offsets)={len(args.context_offsets)} but max_N={max_N} -> expected {2 * max_N + 1}."
        )
    if args.context_offsets[max_N] != 0:
        raise ValueError(
            "Legacy sequential models expect the center element (index max_N) of context_offsets to be 0.\n"
            f"Got context_offsets[{max_N}]={args.context_offsets[max_N]}."
        )

    decoders = AttributeMetadataDecoder.load(
        metadata_dir=md_dir,
        attribute_value_mapping_path=args.attribute_value_mapping_path,
        expected_attribute_names=ds.info.attribute_names,
    )
    attrs_to_include = set(get_attrs_to_include())

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build per-attribute legacy models
    attribute_to_n_classes = (
        config["filtered_attribute_to_n_classes"]
        if "filtered_attribute_to_n_classes" in config
        else config["attribute_to_n_classes"]
    )
    attribute_to_input_type_to_metadata = config["attribute_to_input_type_to_metadata"]
    attribute_to_lstm_hyperparams = config["attribute_to_lstm_hyperparams"]

    attribute_to_checkpoint = _build_attribute_to_checkpoint_legacy(config, args.seq_models_root)

    models: dict[str, LegacyGeneralLSTMModel] = {}
    for attribute, n_classes in attribute_to_n_classes.items():
        input_type_to_metadata = attribute_to_input_type_to_metadata[attribute]
        lstm_hparams = attribute_to_lstm_hyperparams[attribute]
        m = LegacyGeneralLSTMModel(
            max_N=max_N,
            N=N,
            n_classes=int(n_classes),
            input_type_to_metadata=input_type_to_metadata,
            lstm_hyperparams=lstm_hparams,
        )
        ckpt_path = attribute_to_checkpoint.get(attribute)
        if ckpt_path is None:
            continue
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Sequential checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Expected 'model_state_dict' in sequential checkpoint: {ckpt_path}")
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m.to(device)
        m.eval()
        models[attribute] = m

    # Iterate BiH batches, but run sequential models on feature sequences.
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    n_done = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"sequential_legacy:{args.split}"):
            rgb = batch["rgb"]  # keep on CPU for visualization
            seg_ids = batch["segment_id"]

            for i in range(rgb.shape[0]):
                sid = str(seg_ids[i])
                context_ids = list(ds.segment_id_to_context_ids[sid])
                feats_seq = _load_feat_sequence(args.feat_dir, context_ids).unsqueeze(0).to(device)  # (1, S, D)

                lines: list[str] = []
                for attribute, m in models.items():
                    if attribute not in attrs_to_include:
                        continue
                    logits = m({"feats": feats_seq})
                    pred_idx = int(logits.argmax(dim=-1).item())
                    lines.append(decoders.to_text(attr=attribute, class_idx=pred_idx))

                out_path = save_inference_visualization(
                    rgb_seq=rgb[i].detach().cpu(),
                    text="\n".join(lines),
                    segment_id=sid,
                    output_dir=args.output_dir,
                    out_size=(args.out_width, args.out_height),
                    text_area_ratio=args.text_area_ratio,
                )
                n_done += 1
                if args.limit is not None and n_done >= args.limit:
                    if args.verbose:
                        print(f"Reached --limit={args.limit}. Last output: {out_path}")
                    return

    if args.verbose:
        print(f"Saved {n_done} visualizations to {args.output_dir}")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="IRAP BiH inference + visualization (vidlu_irap_gaim)")
    p.add_argument("--mode", choices=("local", "sequential_legacy"), default="local")
    p.add_argument("--dataset_dir", type=str, default=None, help="Path to IRAP_BIH (optional; else from IRAP_HOME)")
    p.add_argument(
        "--metadata_dir", type=str, default=None, help="Path to IRAP_BIH_METADATA (optional; else from IRAP_HOME)"
    )
    p.add_argument(
        "--attribute_value_mapping_path",
        type=str,
        default=None,
        help="Optional value mapping JSON (must match dataset build)",
    )
    p.add_argument(
        "--seg_to_res_path", type=str, default=None, help="Optional seg_to_res directory for n-context filtering"
    )
    p.add_argument("--no_ncontext_filter", action="store_true", help="Disable N-context filtering")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument(
        "--context_offsets",
        type=_parse_int_list_csv,
        default=(0, -1, -4),
        help="CSV list of integer offsets, e.g. '0,-1,-4'",
    )

    p.add_argument("--output_dir", type=str, default="visualization_output")
    p.add_argument("--out_width", type=int, default=1920)
    p.add_argument("--out_height", type=int, default=1080)
    p.add_argument("--text_area_ratio", type=float, default=0.35)

    p.add_argument("--device", type=str, default=None, help="cuda:0 / cpu (default: auto)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Stop after writing this many images")
    p.add_argument("--verbose", action="store_true")

    # Local model args
    p.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Vidlu checkpoint directory containing model_state.pth"
    )
    p.add_argument("--model_state_path", type=str, default=None, help="Direct path to a model state dict (.pth/.pt)")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--pretrained_backbone", action="store_true", help="Use ImageNet pretrained ResNet18 backbone")

    # Input adapter emulation (match training)
    p.add_argument("--input_adapter", choices=("id", "standardize"), default="id")
    p.add_argument("--mean", type=float, nargs=3, default=RGB_MEAN)
    p.add_argument("--std", type=float, nargs=3, default=RGB_STD)

    # Sequential legacy args
    p.add_argument("--seq_config_path", type=str, default=None, help="Legacy sequential config.json path")
    p.add_argument(
        "--seq_models_root", type=str, default=None, help="Root directory containing sequential model subdirs"
    )
    p.add_argument("--feat_dir", type=str, default=None, help="Directory with exported per-segment feature .npy files")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "local":
        run_local(args)
    elif args.mode == "sequential_legacy":
        run_sequential_legacy(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
