import os

import torch
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import MultiplicativeLR
from collections.abc import Mapping

from vidlu.training.extensions import TrainerExtension
from vidlu.utils.collections import NameDict


def phase_lr_decay_factor(total_decay: float, num_epochs: int) -> float:
    """Per-epoch multiplicative LR factor that decays by `total_decay` over
    `num_epochs` epochs."""
    return total_decay ** (1 / num_epochs)


class FreezeThenFinetune(TrainerExtension):
    """Two-phase training: heads-only ("frozen") epochs followed by full-model finetuning,
    each phase with a fresh Adam and an exponential LR decay.

    What is fixed per phase is the *total* decay, not the per-epoch factor, so that changing
    `epoch_count` does not change the LR the phase ends at. The defaults are the original
    recipe's totals (2 epochs at 0.8/epoch, 13 at 0.88/epoch), so `epoch_count=15` reproduces
    it exactly.
    """

    def __init__(
        self,
        num_frozen_epochs: int = 2,
        frozen_lr: float = 5e-5,
        finetune_lr: float = 1e-5,
        frozen_weight_decay: float = 1e-3,
        finetune_weight_decay: float = 1e-3,
        frozen_lr_total_decay: float = 0.8 ** 2,
        finetune_lr_total_decay: float = 0.88 ** 13,
    ):
        # Phase lengths (the finetune phase spans the trainer's remaining epochs)
        self.num_frozen_epochs = num_frozen_epochs

        # Optimizer / scheduler hyperparameters per phase
        self.frozen_lr = frozen_lr
        self.finetune_lr = finetune_lr
        self.frozen_wd = frozen_weight_decay
        self.finetune_wd = finetune_weight_decay
        self.frozen_lr_total_decay = frozen_lr_total_decay
        self.finetune_lr_total_decay = finetune_lr_total_decay

    def initialize(self, trainer):
        num_finetune_epochs = trainer.epoch_count - self.num_frozen_epochs
        if num_finetune_epochs <= 0:
            raise ValueError(
                f"FreezeThenFinetune requires epoch_count > num_frozen_epochs, got"
                f" epoch_count={trainer.epoch_count}, num_frozen_epochs={self.num_frozen_epochs}.")

        def reinit_optimizer_and_scheduler(lr, wd, total_decay, num_phase_epochs):
            # preserve optimizer state? start fresh as in original code
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=lr,
                weight_decay=wd,
            )
            factor = phase_lr_decay_factor(total_decay, num_phase_epochs)
            trainer.optimizer = opt
            trainer.lr_scheduler = MultiplicativeLR(opt, lr_lambda=lambda epoch: factor)

        def get_trainable_parameters():
            """Get trainable parameters following original logic: use model's method if available"""
            if hasattr(trainer.model, "get_trainable_parameters"):
                return trainer.model.get_trainable_parameters()
            else:
                # Fallback: heads + SPP + attention for frozen phase
                trainable_params = []
                heads = getattr(trainer.model, "heads", [])
                for head in heads:
                    trainable_params.extend(head.parameters())
                attn_blocks = getattr(trainer.model, "attn_blocks", None)
                if attn_blocks is not None:
                    trainable_params.extend(attn_blocks)
                spp = getattr(trainer.model, "spp", None)
                if spp is not None:
                    trainable_params.extend(spp.parameters())
                return trainable_params

        def set_phase(phase: str):
            # Freeze/unfreeze parameters following original logic
            if phase == "frozen":
                # First freeze all parameters
                for p in trainer.model.parameters():
                    p.requires_grad = False
                # Then enable only the trainable subset (heads + SPP)
                trainable_params = get_trainable_parameters()
                for p in trainable_params:
                    p.requires_grad = True
                reinit_optimizer_and_scheduler(self.frozen_lr, self.frozen_wd,
                                               self.frozen_lr_total_decay, self.num_frozen_epochs)
            else:  # finetune
                # Enable all parameters in finetune phase
                for p in trainer.model.parameters():
                    p.requires_grad = True
                reinit_optimizer_and_scheduler(self.finetune_lr, self.finetune_wd,
                                               self.finetune_lr_total_decay, num_finetune_epochs)

        # Initialize in frozen phase at start
        set_phase("frozen")

        # Handle resumption: ensure phase is correct before loading optimizer state
        original_load_state_dict = trainer.load_state_dict

        def load_state_dict_wrapper(state_dict):
            # Check the epoch in the checkpoint
            # state_dict['training'] is the EpochLoop state dict, containing 'epoch'
            if "training" in state_dict and "epoch" in state_dict["training"]:
                epoch = state_dict["training"]["epoch"]
                # If we finished the frozen epochs, we should be in finetune phase
                # (e.g. if num_frozen_epochs=2, epochs 0 and 1 are frozen.
                #  If we save at end of epoch 1, stored epoch is 1. Next is 2.
                #  Transition happens at START of epoch 2.
                #  So if stored epoch >= 2, we are definitely in finetune.
                #  Wait, if stored epoch is 1, we are at end of frozen. Optimizer is frozen.
                #  If stored epoch is 2, we are at end of first finetune epoch. Optimizer is finetune.
                if epoch >= self.num_frozen_epochs:
                    set_phase("finetune")
                else:
                    set_phase("frozen")

            original_load_state_dict(state_dict)

        trainer.load_state_dict = load_state_dict_wrapper

        @trainer.training.epoch_started.handler
        def on_epoch_started(state):
            if state.epoch == self.num_frozen_epochs:
                set_phase("finetune")


class MultiAttributeScorePrinter(TrainerExtension):
    """
    Prints metrics to stdout after evaluation completes, with special formatting
    for per-attribute metrics (nested dictionaries).
    """

    def initialize(self, trainer):
        @trainer.evaluation.completed.handler
        def on_eval_completed(state):
            if not hasattr(state, "metrics"):
                return

            per_attr_metrics = {name : value for name, value in state.metrics.items() if isinstance(value, Mapping)}

            if not per_attr_metrics:
                return

            print("\nPer-Attribute Metrics:")

            # Format as Table
            attrs = list(dict.fromkeys(
                attr for m_dict in per_attr_metrics.values() for attr in m_dict
            ))

            # Columns: Attribute, Metric1, Metric2...
            # Metric headers: Display name (lstrip _)
            metric_names = sorted(per_attr_metrics.keys(), key=lambda x: x.lstrip("_"))
            headers = ["Attribute"] + [m.lstrip("_") for m in metric_names]

            # Determine column widths
            col_widths = [len(h) for h in headers]

            # Helper to get formatted value
            def get_val_str(m_name, attr):
                if m_name not in per_attr_metrics or attr not in per_attr_metrics[m_name]:
                    return "-"
                val = per_attr_metrics[m_name][attr]
                if isinstance(val, (int, np.integer)):
                    return str(int(val))
                try:
                    fval = float(val)
                    return f"{fval:.4f}"
                except Exception:
                    return str(val)

            # Pre-calculate rows to determine widths
            rows = []
            for attr in attrs:
                row = [str(attr)]
                col_widths[0] = max(col_widths[0], len(str(attr)))

                for i, m_name in enumerate(metric_names):
                    s = get_val_str(m_name, attr)
                    row.append(s)
                    col_widths[i + 1] = max(col_widths[i + 1], len(s))
                rows.append(row)

            # Print Header
            # Attribute (string) left align. Numbers right align.
            header_strs = []
            header_strs.append(headers[0].ljust(col_widths[0]))
            for i in range(1, len(headers)):
                header_strs.append(headers[i].rjust(col_widths[i]))

            print("  " + "  ".join(header_strs))
            print("  " + "  ".join(["-" * w for w in col_widths]))

            # Print Rows
            for row in rows:
                row_strs = []
                row_strs.append(row[0].ljust(col_widths[0]))
                for i in range(1, len(row)):
                    row_strs.append(row[i].rjust(col_widths[i]))
                print("  " + "  ".join(row_strs))

            print("\n")


class VisualizationExtension(TrainerExtension):
    def __init__(self, dirs, debug_dir_name="debug_vis/vidlu"):
        self.dirs = dirs
        self.debug_dir = os.path.join(os.path.dirname(dirs.experiments), debug_dir_name)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.visualized = False

    def initialize(self, trainer):
        @trainer.training.iter_completed.handler
        def on_iter_completed(state):
            if self.visualized or os.environ.get("IRAP_DEBUG") != "1":
                return

            # Only visualize the first iteration of the first epoch
            if state.epoch > 0 or state.iteration > 0:
                return

            print(f"Visualizing batch to {self.debug_dir}...")

            # Extract data
            batch = state.batch
            result = state.result

            # 1. Visualize Inputs (Images)
            # Assuming batch is a dict or NameDict with 'rgb' key or similar, or x is in result
            x = result.get("x")
            if x is not None:
                # x shape: (B, C, H, W) or (B, S, C, H, W) for sequences
                if x.ndim == 5:  # Sequence (B, S, C, H, W)
                    # Visualize first frame of first few samples
                    imgs = x[:, 0]
                else:
                    imgs = x

                # Denormalize if necessary (assuming standard normalization or 0-1)
                # Here we assume 0-1 range for simplicity as per input_adapter="id" in command
                # If "standardize" was used, we might need to revert it, but "id" implies identity.

                # Log input statistics
                with open(os.path.join(self.debug_dir, "input_stats.txt"), "w") as f:
                    f.write(f"Input shape: {imgs.shape}\n")
                    for i in range(min(imgs.shape[0], 8)):
                        img_tensor = imgs[i].detach().cpu()
                        f.write(
                            f"Image {i} - Min: {img_tensor.min().item()}, Max: {img_tensor.max().item()}, Mean: {img_tensor.mean().item()}, Std: {img_tensor.std().item()}\n"
                        )

                # BIH mean and std
                from irap_data.irap_dataset import RGB_MEAN, RGB_STD

                mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
                std = torch.tensor(RGB_STD).view(3, 1, 1)

                for i in range(min(imgs.shape[0], 8)):  # Save up to 8 images
                    img_tensor = imgs[i].detach().cpu()

                    # Denormalize
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)

                    # (C, H, W) -> (H, W, C)
                    img_np = img_tensor.permute(1, 2, 0).numpy()

                    img_np = (img_np * 255).astype(np.uint8)

                    try:
                        Image.fromarray(img_np).save(os.path.join(self.debug_dir, f"input_{i}.png"))
                    except Exception as e:
                        print(f"Failed to save image {i}: {e}")

            # 2. Visualize Targets and Predictions
            target = result.get("target")
            out = result.get("out")

            if target is not None and out is not None:
                with open(os.path.join(self.debug_dir, "predictions.txt"), "w") as f:
                    f.write(f"Batch size: {target.shape[0]}\n")

                    # Assuming out is logits (B, A, K) or similar, and target is (B, A)
                    # Check shapes
                    if isinstance(out, (list, tuple)):
                        f.write(f"Output is a {type(out)} of length {len(out)}\n")
                        for i, o in enumerate(out):
                            f.write(f"Output {i} shape: {o.shape}\n")
                            preds = o.argmax(dim=-1)
                            f.write(f"Predictions {i} (argmax):\n{preds.detach().cpu().numpy()}\n")

                            # Target handling
                            if target.ndim == 2 and target.shape[1] == len(out):
                                t = target[:, i]
                                f.write(f"Targets {i}:\n{t.detach().cpu().numpy()}\n")
                            else:
                                f.write(f"Targets (raw):\n{target.detach().cpu().numpy()}\n")
                    else:
                        f.write(f"Output shape: {out.shape}\n")
                        if out.ndim == 3:  # (B, A, K) - Multi-attribute classification
                            preds = out.argmax(dim=-1)  # (B, A)
                            f.write(f"Predictions (argmax):\n{preds.detach().cpu().numpy()}\n")
                            f.write(f"Targets:\n{target.detach().cpu().numpy()}\n")
                        else:
                            f.write(f"Output raw:\n{out.detach().cpu().numpy()}\n")
                            f.write(f"Targets raw:\n{target.detach().cpu().numpy()}\n")

            # 3. Visualize Loss Weights
            if hasattr(trainer, "loss") and hasattr(trainer.loss, "_attr_idx_to_class_weights"):
                weights = trainer.loss._attr_idx_to_class_weights
                with open(os.path.join(self.debug_dir, "loss_weights.txt"), "w") as f:
                    if weights:
                        for attr_idx, w in weights.items():
                            f.write(f"Attribute {attr_idx} weights:\n{w.detach().cpu().numpy()}\n")
                    else:
                        f.write("No class weights set in trainer.loss._attr_idx_to_class_weights\n")
            else:
                print("Trainer loss does not have _attr_idx_to_class_weights attribute.")

            self.visualized = True
