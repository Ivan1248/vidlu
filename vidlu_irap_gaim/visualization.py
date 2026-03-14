import os
import torch
import numpy as np
from PIL import Image
from vidlu.training.extensions import TrainerExtension
from vidlu.utils.collections import NameDict
from .datasets import RGB_MEAN, RGB_STD


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
