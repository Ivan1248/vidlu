import cv2
import numpy as np

from vidlu.utils.func import vectorize
import torch


# NumPy ############################################################################################

def numpy_segmentation_distance_transform_single(segmentation, class_count=None):
    present_classes = np.unique(segmentation)
    if class_count is None:
        class_count = present_classes[-1]
    distances = np.full([class_count] + list(segmentation.shape), -1, dtype=np.float32)
    for i in present_classes if present_classes[0] >= 0 else present_classes[1:]:
        class_mask = segmentation == i
        distances[i][class_mask] = cv2.distanceTransform(
            np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
    return distances


# Torch ############################################################################################
# layout: CHW

@vectorize
def hwc_to_chw(x):
    return x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)


class HWCToCHW:
    __call__ = staticmethod(hwc_to_chw)  # keywords: call, copy, ...


@vectorize
def chw_to_hwc(x):
    return x.permute(1, 2, 0) if len(x.shape) == 3 else x.permute(0, 2, 3, 1)


class CHWToHWC:
    __call__ = staticmethod(chw_to_hwc)  # keywords: call, copy, ...


def segmentation_distance_transform(segmentations, class_count=None, dtype=None):
    dts = [numpy_segmentation_distance_transform_single(seg.cpu().numpy(), class_count)
           for seg in segmentations]
    return torch.tensor(dts, device=segmentations.device, dtype=dtype)
