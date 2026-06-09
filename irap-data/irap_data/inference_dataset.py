from types import SimpleNamespace
import typing as T
from pathlib import Path

import numpy as np
import torch

from .dataset import Dataset

from .irap_dataset import RGB_MEAN, RGB_STD, INPUT_DIM
from .image_utils import load_image_cv2, rgb_to_chw_tensor


class InferenceImageDataset(Dataset):
    """
    Dataset for running inference on arbitrary images with temporal context.

    Unlike IRAPDataset, this dataset:
    - Does NOT require labels (for pure inference without ground truth)
    - Uses alphabetical order of images to determine temporal context
    - Borrows attribute metadata from a reference dataset or explicit config

    Context sequence example:
        context_offsets=(0, -1, -4) means for image at index i, load images at i, i-1, i-4.
        Images without valid context (near the start) are excluded.

    Example usage:
        # From folder with context matching IRAPDataset
        ds = InferenceImageDataset.from_folder(
            "/path/to/images",
            reference_dataset=bih_test,
            context_offsets=(0, -1, -4),
        )
    """

    def __init__(
        self,
        image_paths: T.Sequence[Path],
        *,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        class_counts: tuple[int, ...],
        context_offsets: T.Sequence[int] = (0, -1, -4),
        mean: T.Sequence[float] = RGB_MEAN,
        std: T.Sequence[float] = RGB_STD,
        input_size: tuple[int, int] = (INPUT_DIM[0], INPUT_DIM[1]),  # (W, H)
    ):
        """
        Args:
            image_paths: Ordered sequence of image paths (alphabetical order assumed).
            attr_to_value_to_class_idx: Attribute metadata for decoding predictions.
            class_counts: Number of classes per attribute.
            context_offsets: Offsets for context frames, e.g., (0, -1, -4).
                The frames are ordered by offset (so (0, -1, -4) gives [img_0, img_-1, img_-4]).
            mean, std: Normalization stats (used by input adapter, stored in info).
            input_size: Target image size (W, H).
        """
        all_image_paths = [Path(p) for p in image_paths]
        self.context_offsets = tuple(context_offsets)
        self.input_size = input_size

        # Determine which images have valid context (all context indices in bounds)
        min_offset = min(self.context_offsets)
        max_offset = max(self.context_offsets)
        num_images = len(all_image_paths)

        # Valid indices: those where idx + min_offset >= 0 and idx + max_offset < num_images
        self.valid_indices: list[int] = []
        for idx in range(num_images):
            if idx + min_offset >= 0 and idx + max_offset < num_images:
                self.valid_indices.append(idx)

        self.all_image_paths = all_image_paths

        super().__init__(
            subset="inference",
            info=dict(
                problem="multi_attribute_classification",
                class_counts=class_counts,
                pixel_stats=SimpleNamespace(mean=np.array(mean), std=np.array(std)),
                attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            ),
        )

        print(
            f"[InferenceImageDataset] {len(all_image_paths)} images, "
            f"{len(self.valid_indices)} with valid context (sequence={self.context_offsets})"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def get_example(self, idx: int) -> dict:
        img_idx = self.valid_indices[idx]
        path = self.all_image_paths[img_idx]

        frames = [
            rgb_to_chw_tensor(load_image_cv2(str(self.all_image_paths[img_idx + offset])),
                              self.input_size)
            for offset in self.context_offsets
        ]
        rgb = torch.stack(frames, dim=0)

        return dict(rgb=rgb, segment_id=path.stem)

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        reference_dataset=None,
        attr_to_value_to_class_idx: dict | None = None,
        class_counts: tuple[int, ...] | None = None,
        context_offsets: T.Sequence[int] = (0, -1, -4),
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        **kwargs,
    ) -> "InferenceImageDataset":
        """
        Create dataset from a folder of images.

        Images are sorted alphabetically to determine temporal order.
        Only images with valid context (all context indices in bounds) are included.

        Args:
            folder: Path to folder containing images.
            reference_dataset: A IRAPDataset to copy metadata from (alternative to explicit args).
            attr_to_value_to_class_idx: Explicit attribute metadata (if no reference_dataset).
            class_counts: Explicit class counts (if no reference_dataset).
            context_offsets: Offsets for context frames, e.g., (0, -1, -4).
            extensions: Image file extensions to include.
            **kwargs: Additional arguments passed to __init__.
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        # Collect image paths in alphabetical order
        image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ])
        if not image_paths:
            raise ValueError(f"No images with extensions {extensions} found in {folder}")

        # Get metadata
        if reference_dataset is not None:
            attr_to_value_to_class_idx = reference_dataset.info.attr_to_value_to_class_idx
            class_counts = reference_dataset.info.class_counts
        elif attr_to_value_to_class_idx is None or class_counts is None:
            raise ValueError(
                "Either reference_dataset or both attr_to_value_to_class_idx and class_counts must be provided"
            )

        return cls(
            image_paths,
            attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            class_counts=class_counts,
            context_offsets=context_offsets,
            **kwargs,
        )
