from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

from vidlu.utils.loadsave import NumpyLoadSave, PickleLoadSave

class_info_to_fmgr = dict(incidence=NumpyLoadSave,
                          instances=PickleLoadSave,
                          dist=NumpyLoadSave)


def get_class_segment_boxes(y, classes):
    image_instances = {c: [] for c in classes}
    for c in classes:
        mask = np.uint8(y == c)
        num, masks = cv2.connectedComponents(mask)
        for j in range(1, num):
            component = masks == j
            cols = np.where(np.any(component, axis=0))[0]  # W
            rows = np.where(np.any(component, axis=1))[0]  # H
            cols_min, cols_max = cols.min().item(), cols.max().item()
            rows_min, rows_max = rows.min().item(), rows.max().item()
            image_instances[c] += [[cols_min, cols_max, rows_min, rows_max]]
    return {c: np.array(v, dtype=np.uint16) for c, v in image_instances.items()}


def segmentation_class_info(ds):
    C = ds.info.class_count
    class_dist = np.zeros((len(ds), C + 1), dtype=np.uint64)
    instances = []
    for i, example in enumerate(tqdm(ds, total=len(ds))):
        y = example['y'].squeeze().numpy()
        classes, counts = np.unique(y, return_counts=True)
        class_dist[i, classes] += counts
        instances.append(get_class_segment_boxes(y, classes))
    incidence = class_dist.sum(0)
    result = dict(incidence=incidence, instances=instances, dist=class_dist)
    return result


def save_class_info(class_info, subset, suffix, ds_root):
    for k, v in class_info.items():
        class_info_to_fmgr[k].save(ds_root / f'class_{k}_{subset}{suffix}', v)


def save_class_infos(subset_to_class_info, suffix, ds_root):
    for subset, class_info in subset_to_class_info.items():
        save_class_info(class_info, subset, suffix, ds_root)


def load_class_info(subset, suffix, ds_root):
    return {k: class_info_to_fmgr[k].load(ds_root / f'class_{k}_{subset}{suffix}')
            for k in tqdm(class_info_to_fmgr.keys(), subset)}


# from util
def find_rare_indices(instances):
    # instances = pickle.load(f)
    class_to_indices = defaultdict(list)
    for i, (k, v) in enumerate(instances.items()):
        for c in v:
            class_to_indices[c].append(i)
    return class_to_indices
