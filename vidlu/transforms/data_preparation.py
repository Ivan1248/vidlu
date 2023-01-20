import typing as T

import numpy as np
import torch

import vidlu.transforms.image as vti
import vidlu.data.types as dt


def prepare_element(x, key_or_type):
    dom = dt.from_key.get(key_or_type, None) if isinstance(key_or_type, str) else key_or_type
    if dom is None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x
    if dom is dt.Image:
        x = vti.hwc_to_chw(vti.to_torch(x)).to(dtype=torch.float) / 255
    elif dom in (dt.ClassLabel, dt.SegMap):
        x = torch.tensor(x, dtype=torch.int64)  # the loss supports only int64
    return dom(x)


# Based on Mask2Former code: Copyright (c) Facebook, Inc. and its affiliates.
class PaddedImageBatch:
    """A tensor of images of possibly varying sizes (padded) together with the sizes."""

    def __init__(self, images, sizes):
        self.images = images
        self.sizes = sizes

    def get_mask(self, pos_value=1.0, neg_value=0.0):
        mask = torch.full_like(self.images, neg_value)
        for i, size in enumerate(self.sizes):
            mask[i, ..., :size[0], :size[1]] = pos_value

    def to_list(self):
        return [self[i, :, :size[0], :size[1]] for i, size in enumerate(self.sizes)]

    @staticmethod
    def from_list(images: T.List[torch.Tensor], size_divisibility: int = 0,
                  pad_value: float = 0.0) -> 'PaddedImageBatch':
        assert isinstance(images, (tuple, list))

        image_sizes = torch.tensor([list(x.shape[-2:]) for x in images])
        max_size = image_sizes.max(0).values
        min_size = image_sizes.max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        if tuple(max_size) == tuple(min_size):
            batched_images = torch.stack(images)
        else:
            batch_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
            batched_images = images[0].new_full(batch_shape, pad_value, device=images[0].device)
            for i, img in enumerate(images):
                batched_images[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)

        return PaddedImageBatch(batched_images, image_sizes)
