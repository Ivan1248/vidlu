from warnings import warn

import numpy as np
import torch
from tqdm import tqdm, trange

from vidlu.utils.func import partial
from vidlu.data import Dataset, Record, class_mapping


def rotating_labels(ds: Dataset) -> Dataset:
    """Orders examples so that in each length C slice of the dataset all C
    classes are present.

    In each slice classes start from 1 to C
    """
    class_count = ds.info['class_count']
    # Divide examples into groups by class
    class_subset_indices = ds.find_indices(
        [lambda d, i=i: bool(d.class_label == i) for i in range(class_count)],
        progress_bar=partial(tqdm, desc='rotating_labels'))[:-1]
    if any(len(csi) != len(ds) // class_count for csi in class_subset_indices):
        raise ValueError(f"The distribution of labels in the dataset should be uniform, "
                         f"not {[len(csi) for csi in class_subset_indices]}.")
    # rearrange examples so that ds[i].y = i mod class_count
    indices = [class_subset_indices[i % class_count][i // class_count] for i in trange(len(ds))]
    return ds[indices]


def chunk(ds: Dataset, size, i) -> Dataset:
    if 0 < size < 1:
        size = int(len(ds) * size + 0.5)
    elif isinstance(size, float):
        raise RuntimeError(f"Invalid argument type: size: {type(size)} = {size}.")
    chunk = ds[i * size:(i + 1) * size]
    if len(chunk) != size:
        raise RuntimeError(f"The fold with index {i} can only have size {len(chunk)} < {size}.")
    return chunk


def folds(ds: Dataset, n: int):
    indices = np.linspace(0.5, len(ds) + 0.5, n + 1, dtype=int)

    def get_fold_f(i):
        return lambda: ds[indices[i]:indices[i + 1]]

    return Record({f"{i}_": get_fold_f(i) for i in range(n)})


def chunks(ds: Dataset, size: int):
    if len(ds) % size != 0:
        warn(f"Dataset size ({len(ds)}) is not divisible by chunk size ({size}).")
    indices = [x * size for x in range(len(ds) // size)]

    def get_fold_f(i):
        return lambda: ds[indices[i]:indices[i + 1]]

    return Record({f"{i}_": get_fold_f(i) for i in range(len(indices) - 1)})


def remap_classes(ds: Dataset, mapping, label_key=None, additional_info=None,
                  additional_fields=None):
    from vidlu.transforms.numpy import remap_segmentation

    additional_info = additional_info or {}
    additional_fields = additional_fields or {}

    if label_key is None:
        if 'seg_map' in ds[0].keys():
            label_key = 'seg_map'
        else:
            raise NotImplementedError("No supported label field found.")

    enc_mapping = class_mapping.encode_many_to_one_mapping(
        mapping, value_to_ind=list({v: None for v in mapping.values()}))

    def convert(label):
        if is_torch := isinstance(label, torch.Tensor):
            label = label.numpy()
        label = remap_segmentation(label, enc_mapping)
        return torch.from_numpy(label).to(device=label.device) if is_torch else label

    return ds.map(lambda r: type(r)(r, **{label_key: convert(r[label_key])}, **additional_fields),
                  info={**ds.info, **dict(class_count=len(mapping), class_names=list(mapping)),
                        **additional_info})


def add_class_mapping(ds: Dataset, mapping):
    enc_mapping = class_mapping.encode_one_to_many_mapping(
        mapping, value_to_ind={k: i for i, k in enumerate(ds.info.class_names)})

    return ds.map(lambda r: type(r)(r, class_mapping=enc_mapping),
                  info=dict(ds.info, class_mapping=mapping))
