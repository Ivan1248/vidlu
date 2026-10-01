"""
Inference + visualization utility for IRAP BH sequences.

This is a vidlu-native port of `libs/irap_gaim-main/inference_visualization.py`:
- Uses `irap_data.make_bh_data` (instead of DatasetWrapper/ImageSequenceDataset).
- Uses `vidlu_irap_gaim.models.classification.ImageSequenceClassifier`.
- Loads checkpoints from Vidlu's `CheckpointManager` format (supports `model_state.pth`).

The script writes per-segment images combining:
- the context RGB frames
- a text panel with predicted attribute values
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as a script from the repository root without installing `vidlu_irap_gaim`.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from irap_data.attrs import get_attrs_to_include  # noqa: E402
from irap_data import make_bh_data  # noqa: E402
from vidlu_irap_gaim.models.classification import ImageSequenceClassifier  # noqa: E402
from vidlu_irap_gaim.models.encoders.resnet import ResNetEncoder  # noqa: E402
from vidlu_irap_gaim.tools.vis_utils import save_inference_visualization  # noqa: E402
from vidlu_irap_gaim.vlm.response_parser import build_idx_to_value  # noqa: E402


def _parse_int_list_csv(x: str) -> tuple[int, ...]:
    if x.strip() == "":
        return tuple()
    return tuple(int(s.strip()) for s in x.split(",") if s.strip() != "")


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


def run(args) -> None:
    datasets = make_bh_data(
        dataset_dir=args.dataset_dir,
        metadata_dir=args.metadata_dir,
        context_offsets=args.context_offsets,
        use_ncontext_filter=not args.no_ncontext_filter,
    )
    ds = datasets[args.split]
    idx_to_value = build_idx_to_value(ds.info.attr_to_value_to_class_idx)

    attrs_to_include = set(get_attrs_to_include())
    attribute_names = list(ds.info.attr_to_value_to_class_idx)
    attribute_to_idx = {a: i for i, a in enumerate(attribute_names)}

    # Model: match training defaults (sequence_length = len(context_offsets))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageSequenceClassifier(
        class_counts=tuple(ds.info.class_counts),
        attention=args.attention,
        sequence_length=len(args.context_offsets),
        encoder_f=lambda: ResNetEncoder(pretrained=args.pretrained_backbone,
                                        pixel_stats=ds.info.pixel_stats),
    ).to(device)

    # DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # The heads are built on the first call, so the checkpoint can only be loaded after one.
    model.eval()
    with torch.no_grad():
        model({"rgb": next(iter(dl))["rgb"].to(device)})

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

    n_done = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc=args.split):
            rgb = batch["rgb"].to(device)
            seg_ids = batch["segment_id"]

            # The encoder normalizes internally, so it takes [0, 1] frames as loaded.
            outputs = model({"rgb": rgb})  # tuple of (B, K_i)

            for i in range(rgb.shape[0]):
                lines: list[str] = []
                for attr in attribute_names:
                    if attr not in attrs_to_include:
                        continue
                    aidx = attribute_to_idx[attr]
                    pred_idx = int(outputs[aidx][i].argmax(dim=-1).item())
                    lines.append(f"{attr}: {idx_to_value[attr][pred_idx]} ({pred_idx})")
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


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="IRAP BH inference + visualization (vidlu_irap_gaim)")
    p.add_argument("--dataset_dir", type=str, default=None, help="Path to IRAP_BIH (optional; else from IRAP_HOME)")
    p.add_argument(
        "--metadata_dir", type=str, default=None, help="Path to IRAP_BIH_METADATA (optional; else from IRAP_HOME)"
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

    # Model args
    p.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Vidlu checkpoint directory containing model_state.pth"
    )
    p.add_argument("--model_state_path", type=str, default=None, help="Direct path to a model state dict (.pth/.pt)")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--pretrained_backbone", action="store_true", help="Use ImageNet pretrained ResNet18 backbone")

    # Normalization is the encoder's own (see models/encoders/base.py), so there is
    # nothing to match here – applying it again would double-normalize.

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
