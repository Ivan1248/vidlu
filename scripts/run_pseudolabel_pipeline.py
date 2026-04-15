#!/usr/bin/env python3
"""Semi-supervised pseudo-label pipeline.

Orchestrates three phases of pseudo-label self-training by calling run.py as
subprocesses. All data/model/trainer strings mirror run.py's positional
argument convention and default to IRAP BiH values.

Usage examples:
  # Full pipeline (on-the-fly), IRAP defaults:
  python run_pseudolabel_pipeline.py

  # Only supervised phase:
  python run_pseudolabel_pipeline.py --phases supervised

  # Offline mode with custom labeled ratio in data string:
  python run_pseudolabel_pipeline.py \\
    --data-semisup "irap_gaim.make_semisup_bih_data(labeled_ratio=0.05)" \\
    --data-offline "irap_gaim.make_pseudo_labeled_bih_data(labeled_ratio=0.05,pseudo_labels_path='{pseudo_labels_path}')" \\
    --mode offline

  # Generalising to another dataset (provide all four run.py args):
  python run_pseudolabel_pipeline.py \\
    --data-supervised "MyDataset{train,val}" \\
    --data-semisup "train,train_u,val:MyDataset{train,val}:(*d[0].split(100),d[1])" \\
    --input-adapter id \\
    --model "MyModel" \\
    --trainer-supervised "tc.my_trainer" \\
    --trainer-pseudolabel "tc.my_trainer,train_step=MyPseudoLabelStep(pre_trained_teacher='{teacher_path}')" \\
    --trainer-offline "tc.my_offline_trainer" \\
    --data-offline "MyPseudoLabeledDataset(pseudo_labels_path='{pseudo_labels_path}')" \\
    --params "" --metrics ""
"""

import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RUN_PY = str(SCRIPTS_DIR / "run.py")

ALL_PHASES = ["supervised", "generate", "train-pseudolabel"]

# ── IRAP BiH defaults ─────────────────────────────────────────────────────────
_IRAP_MODEL = (
    "irap_gaim.ImageSequenceClassifier"
    ",class_counts=irap_gaim.get_class_counts()"
    ",attention=False"
    ",sequence_length=3"
    ",encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)"
)
_IRAP_TRAINER_PSEUDOLABEL = (
    "irap_gaim.irap_pseudo_label_trainer"
    ",train_step=irap_gaim.MultiAttributePseudoLabelStep("
    "pre_trained_teacher='{teacher_path}'"
    ",conf_thresh={conf_thresh}"
    ",temperature={temperature})"
)
_IRAP_GENERATE_MODULE = (
    "vidlu_irap_gaim.tools.generate_pseudo_labels"
    ":run_pipeline,e"
    ",output='{pseudo_labels_path}'"
    ",conf_thresh={conf_thresh}"
    ",temperature={temperature}"
)


# ── Template substitution ─────────────────────────────────────────────────────

class _PartialDict(dict):
    """dict that returns '{key}' for missing keys, enabling partial str.format_map."""
    def __missing__(self, key):
        return f'{{{key}}}'


def _fill(template: str, **kwargs) -> str:
    """Fill known placeholders in template; leave unknown ones as-is."""
    return template.format_map(_PartialDict(**kwargs))


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def _get_experiment_dir(data: str, input_adapter: str, model: str, trainer: str,
                        exp_suffix: str, params: str, metrics: str) -> Path:
    """Ask run.py for the experiment directory path via get_path subcommand."""
    cmd = [sys.executable, RUN_PY, "get_path",
           data, input_adapter, model, trainer,
           "-e", exp_suffix]
    if params:
        cmd += ["--params", params]
    if metrics:
        cmd += ["--metrics", metrics]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=os.environ)
    # get_path prints the experiment dir as the last line, e.g.:
    #   /data/experiments/states/<exp_name>
    # Other initialization code may print to stdout before it, so take only the last line.
    return Path(result.stdout.strip().splitlines()[-1])


def _find_best_checkpoint(exp_dir: Path) -> Path:
    """Scan exp_dir for checkpoint subdirectories and return the best model_state.pth."""
    best_perf = None
    best_subdir = None
    for subdir in exp_dir.iterdir():
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.p"
        if not summary_path.exists():
            continue
        try:
            summary = torch.load(summary_path, map_location="cpu", weights_only=False)
            perf = summary.get("perf", None)
            if perf is None:
                continue
            if best_perf is None or perf > best_perf:
                best_perf = perf
                best_subdir = subdir
        except Exception:
            continue
    if best_subdir is None:
        raise RuntimeError(
            f"No valid checkpoints found in {exp_dir}. "
            "Did the 'supervised' phase complete successfully?"
        )
    return best_subdir / "model_state.pth"


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    print("\n[pipeline] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ)


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_supervised(args) -> None:
    cmd = [sys.executable, RUN_PY, "train",
           args.data_supervised, args.input_adapter, args.model, args.trainer_supervised,
           "-d", args.device,
           "-e", f"{args.name}_supervised"]
    if args.params:
        cmd += ["--params", args.params]
    if args.metrics:
        cmd += ["--metrics", args.metrics]
    if args.resume:
        cmd += ["-r", args.resume]
    _run(cmd)


def run_generate(args, pseudo_labels_path: Path) -> None:
    generate_module = _fill(
        args.generate_module,
        pseudo_labels_path=str(pseudo_labels_path),
        conf_thresh=args.conf_thresh,
        temperature=args.temperature,
    )
    pseudo_labels_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, RUN_PY, "test",
           args.data_semisup, args.input_adapter, args.model, args.trainer_supervised,
           "-d", args.device,
           "-e", f"{args.name}_supervised",
           "-r", "best",
           "-m", generate_module]
    if args.params:
        cmd += ["--params", args.params]
    _run(cmd)


def run_train_onthefly(args, teacher_path: Path) -> None:
    trainer = _fill(
        args.trainer_pseudolabel,
        teacher_path=str(teacher_path),
        conf_thresh=args.conf_thresh,
        temperature=args.temperature,
    )
    cmd = [sys.executable, RUN_PY, "train",
           args.data_semisup, args.input_adapter, args.model, trainer,
           "-d", args.device,
           "-e", f"{args.name}_train_pseudolabel_onthefly"]
    if args.params:
        cmd += ["--params", args.params]
    if args.metrics:
        cmd += ["--metrics", args.metrics]
    if args.resume:
        cmd += ["-r", args.resume]
    _run(cmd)


def run_train_offline(args, pseudo_labels_path: Path) -> None:
    data_offline = _fill(args.data_offline, pseudo_labels_path=str(pseudo_labels_path))
    cmd = [sys.executable, RUN_PY, "train",
           data_offline, args.input_adapter, args.model, args.trainer_offline,
           "-d", args.device,
           "-e", f"{args.name}_train_pseudolabel_offline"]
    if args.params:
        cmd += ["--params", args.params]
    if args.metrics:
        cmd += ["--metrics", args.metrics]
    if args.resume:
        cmd += ["-r", args.resume]
    _run(cmd)


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _resolve_pseudo_labels_path(args) -> Path:
    if args.pseudo_labels:
        return Path(args.pseudo_labels)
    # Derive experiments_dir from get_path output
    try:
        exp_dir = _get_experiment_dir(
            args.data_supervised, args.input_adapter, args.model,
            args.trainer_supervised, f"{args.name}_supervised",
            args.params, args.metrics,
        )
        experiments_dir = exp_dir.parent.parent  # strip /<exp_name> and /states
    except Exception:
        experiments_dir = Path(".")
    return experiments_dir / "pseudo_labels" / f"{args.name}.npz"


def _check_supervised_exists(args) -> None:
    try:
        exp_dir = _get_experiment_dir(
            args.data_supervised, args.input_adapter, args.model,
            args.trainer_supervised, f"{args.name}_supervised",
            args.params, args.metrics,
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "ERROR: Could not resolve supervised experiment directory. "
            "Run the 'supervised' phase first or check your --data-supervised / --model / --trainer-supervised args."
        )
    if not exp_dir.exists():
        raise SystemExit(
            f"ERROR: Supervised experiment directory does not exist: {exp_dir}\n"
            "Run --phases supervised first."
        )


def _check_npz_exists(pseudo_labels_path: Path) -> None:
    if not pseudo_labels_path.exists():
        raise SystemExit(
            f"ERROR: Pseudo-labels file not found: {pseudo_labels_path}\n"
            "Run --phases generate first, or pass --pseudo-labels PATH."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # run.py factory string args
    g = parser.add_argument_group("run.py factory strings (mirror run.py positional args)")
    g.add_argument("--data-supervised", default="irap_gaim.make_bih_data()",
                   help="Data factory for the supervised phase")
    g.add_argument("--data-semisup",
                   default="irap_gaim.make_semisup_bih_data(labeled_ratio=0.1)",
                   help="Data factory for generate and on-the-fly train-pseudolabel phases")
    g.add_argument("--data-offline",
                   default="irap_gaim.make_pseudo_labeled_bih_data(labeled_ratio=0.1,pseudo_labels_path='{pseudo_labels_path}')",
                   help="Data factory for offline train-pseudolabel phase; {pseudo_labels_path} is filled by the script")
    g.add_argument("--input-adapter", default="standardize",
                   help="Input adapter string (run.py 2nd positional arg)")
    g.add_argument("--model", default=_IRAP_MODEL,
                   help="Model factory string (run.py 3rd positional arg)")
    g.add_argument("--trainer-supervised", default="irap_gaim.irap_local_rec_trainer",
                   help="Trainer for supervised phase")
    g.add_argument("--trainer-pseudolabel", default=_IRAP_TRAINER_PSEUDOLABEL,
                   help="Trainer template for on-the-fly phase; {teacher_path}, {conf_thresh}, {temperature} are filled")
    g.add_argument("--trainer-offline", default="irap_gaim.irap_pseudo_label_offline_trainer",
                   help="Trainer for offline train-pseudolabel phase")
    g.add_argument("--params",
                   default="id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt",
                   help="--params value passed to run.py (empty string to omit)")  # TODO: un-hard-code irap_gaim/vistas.pt
    g.add_argument("--metrics",
                   default="irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])",
                   help="--metrics value passed to run.py (empty string to omit)")
    g.add_argument("--generate-module", default=_IRAP_GENERATE_MODULE,
                   help="run.py -m value for the generate phase; {pseudo_labels_path}, {conf_thresh}, {temperature} are filled")

    # Pipeline control
    p = parser.add_argument_group("pipeline control")
    p.add_argument("--conf-thresh", type=float, default=0.8,
                   help="Filled into {conf_thresh} placeholder in template strings (default: 0.8)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Filled into {temperature} placeholder in template strings (default: 1.0)")
    p.add_argument("--mode", choices=["onthefly", "offline", "both"], default="onthefly",
                   help="Pseudo-label training mode (default: onthefly)")
    p.add_argument("--phases", nargs="+", default=["all"],
                   metavar="PHASE",
                   help="Phases to run: all, or subset of: supervised generate train-pseudolabel")
    p.add_argument("--name", default="pseudolabel",
                   help="Base name prefix for experiment suffixes (default: pseudolabel)")
    p.add_argument("--device", default="cuda",
                   help="PyTorch device string (default: cuda)")
    p.add_argument("--pseudo-labels", default=None, metavar="PATH",
                   help="Path to existing .npz; skips generate if provided")
    p.add_argument("-r", "--resume", default=None,
                   choices=["?", "strict", "best", "restart"],
                   help="Pass -r <value> to supervised and train-pseudolabel subprocesses")

    args = parser.parse_args()

    # Resolve phases
    if args.phases == ["all"]:
        phases = set(ALL_PHASES)
    else:
        invalid = set(args.phases) - set(ALL_PHASES)
        if invalid:
            raise SystemExit(f"ERROR: Unknown phases: {invalid}. Valid: {ALL_PHASES}")
        phases = set(args.phases)

    # Resolve pseudo-labels path
    pseudo_labels_path = _resolve_pseudo_labels_path(args)

    # Pre-flight checks
    needs_supervised_exp = (
        "generate" in phases
        or ("train-pseudolabel" in phases and args.mode in ("onthefly", "both"))
    )
    if needs_supervised_exp and "supervised" not in phases:
        _check_supervised_exists(args)

    if "train-pseudolabel" in phases and args.mode in ("offline", "both"):
        if "generate" not in phases and not (args.pseudo_labels and Path(args.pseudo_labels).exists()):
            _check_npz_exists(pseudo_labels_path)

    # ── Run phases ────────────────────────────────────────────────────────────

    if "supervised" in phases:
        run_supervised(args)

    if "generate" in phases:
        if args.mode == "onthefly":
            print("[pipeline] Warning: --mode onthefly — skipping generate phase (unused).")
        else:
            if args.pseudo_labels:
                print(f"[pipeline] Warning: --pseudo-labels provided; generate will write to {pseudo_labels_path}")
            run_generate(args, pseudo_labels_path)

    if "train-pseudolabel" in phases:
        if args.mode in ("onthefly", "both"):
            # Discover teacher checkpoint
            exp_dir = _get_experiment_dir(
                args.data_supervised, args.input_adapter, args.model,
                args.trainer_supervised, f"{args.name}_supervised",
                args.params, args.metrics,
            )
            teacher_path = _find_best_checkpoint(exp_dir)
            print(f"[pipeline] Teacher checkpoint: {teacher_path}")
            run_train_onthefly(args, teacher_path)

        if args.mode in ("offline", "both"):
            _check_npz_exists(pseudo_labels_path)
            run_train_offline(args, pseudo_labels_path)


if __name__ == "__main__":
    main()
