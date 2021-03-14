# Taken from https://github.com/Britefury/cutmix-semisup-seg

import numpy as np


class BoxMaskGenerator:
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True,
                 within_bounds=True):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds

    def __call__(self, batch_size, shape, rng=np.random):
        """Generates masks with one or more boxes each.

        The average box aspect ratio equals that of the mask.

        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen(256, (32, 32))

        Args:
            batch_size: Number of masks to generate.
            shape: Mask shape as a `(height, width)` tuple
            rng (otpional): np.random.RandomState instance

        Returns:
            masks: `(N, 1, H, W)` array of masks
        """
        bshape = (batch_size, self.n_boxes)
        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_prop = rng.uniform(*self.prop_range, size=bshape)
            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(0.0, 1.0, size=bshape) * np.log(mask_prop))
                x_props = mask_prop / y_props
            else:
                y_props = x_props = np.sqrt(mask_prop)
            zero_mask = mask_prop == 0.0  # to avoid NaNs
            y_props[zero_mask], x_props[zero_mask] = 0, 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(*self.prop_range, size=bshape)
                x_props = rng.uniform(*self.prop_range, size=bshape)
            else:
                x_props = y_props = rng.uniform(*self.prop_range, size=(batch_size, self.n_boxes))
        max_size = np.array(shape) * np.sqrt(1 / self.n_boxes)
        sizes = np.round(np.stack([y_props, x_props], axis=2) * max_size[None, None, :])

        if self.within_bounds:
            positions = np.round(
                (np.array(shape) - sizes) * rng.uniform(0.0, 1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(shape) * rng.uniform(0.0, 1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        masks = np.ones((batch_size, 1) + shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 0
        return masks
