from collections import defaultdict

from vidlu.data.types import AABB, ClassAABBsOnImage
from vidlu.data.data_loader import SingleDataLoader

import numpy as np
import cv2
from tqdm import tqdm


def aabb_from_mask(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return AABB(min=(y, x), size=(h, w))


def get_segment_masks(mask):
    segment_count, masks = cv2.connectedComponents(mask)
    for i in range(1, segment_count):
        yield masks == i


def segmentation_to_class_aabbs(segmentation, classes=None):
    # Based on code from Marin Oršić.
    if classes is None:
        classes = np.unique(segmentation)
    classes = set(classes)
    class_to_aabbs = {c: [aabb_from_mask(m) for m in get_segment_masks(np.uint8(segmentation == c))]
                      for c in classes}
    return ClassAABBsOnImage(class_to_aabbs, shape=segmentation.shape)


def _example_seg_class_info(example):
    seg = example['seg_map']
    present_classes, counts = np.unique(seg, return_counts=True)
    return dict(classes=present_classes, counts=counts,
                class_aabbs=segmentation_to_class_aabbs(seg, classes=present_classes))


def seg_class_info(ds, num_workers=4):
    # Based on code from Marin Oršić.
    class_freqs = np.zeros((len(ds), ds.info.class_count + 1), dtype=np.uint64)
    class_segment_boxes = []

    ds_infos = ds.map(_example_seg_class_info)
    for i, d in enumerate(tqdm(SingleDataLoader(ds_infos, num_workers=num_workers), total=len(ds),
                               desc="segmentation_class_info")):
        class_freqs[i, d['classes']] += d['counts'].astype(np.uint64)
        class_segment_boxes.append(d['class_aabbs'])
    global_class_freqs = class_freqs.sum(0)
    return dict(class_freqs=class_freqs, class_segment_boxes=class_segment_boxes,
                global_class_freqs=global_class_freqs)


def get_class_to_box_indices(class_segment_boxes):
    result = defaultdict(list)
    for i, (k, v) in enumerate(class_segment_boxes.items()):
        for c in v:
            result[c].append(i)
    return result
