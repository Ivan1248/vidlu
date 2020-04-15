from functools import partial

from tqdm import tqdm, trange

from vidlu.data import Dataset


def rotating_labels(ds: Dataset) -> Dataset:
    class_count = ds.info['class_count']
    class_subset_indices = ds.filter_split_indices(
        [lambda d, i=i: bool(d.y == i)
         for i in range(class_count)], progress_bar=partial(tqdm, desc='rotating_labels'))[:-1]
    if any(len(csi) != len(ds) // class_count for csi in class_subset_indices):
        raise ValueError(f"The distribution of labels in the dataset should be uniform, "
                         f"not {[len(csi) for csi in class_subset_indices]}.")
    indices = [class_subset_indices[i % class_count][i // class_count] for i in trange(len(ds))]
    return ds[indices]
