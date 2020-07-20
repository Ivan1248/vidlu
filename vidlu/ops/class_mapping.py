import typing as T
import torch


def normalize_mapping(mapping: T.Dict[object, int], ignore_key: int = None,
                      ignore_index: int = -1) -> T.Dict[object, int]:
    """ Normalizes an ordered mapping from keys to indices so that indices of
    non-ignored classes start from 0 and increase until `C-1`, where `C` is the
    number of classes, including the "ignore" class.

    Args:
        mapping: An integer-valued ordered mapping. "Ordered" means that the
            order of items is informative, which is important as the remapping
            depends on it (and not on the original indices).
        ignore_key: The key of the "ignore"/"unlabeled" class that is to be
            remapped to `ignore_index`. If it is `None` (default), no class is
            considered to be "ignore".
        ignore_index: The index to ba assigned to `ignore_key`.

    Returns:
        Dict[object, int]
    """
    if ignore_index is None:
        return {c: i for i, c in enumerate(mapping.keys())}
    else:
        result = {c: i for i, c in enumerate(k for k in mapping.keys() if k != ignore_key)}
        result[ignore_key] = ignore_index
        return result


def encode_mapping(mapping: T.Dict[str, T.Iterable[str]], orig_name_to_ind: T.Dict[str, int],
                   univ_name_to_ind: T.Dict[str, int]) -> T.Dict[int, T.Iterable[int]]:
    """Replaces class names with class indices in a 1-to-many class mapping.

    Args:
        mapping: A lists of iterables containing elementary class names.
            `mapping[k]` contains names of all elementary classes that a
            superclass with name `k` comprises.
        orig_name_to_ind: A mapping from superclass names to indices.
        univ_name_to_ind: A mapping from elementary class names to indices.

    Returns:
        Dict[int, Iterable[int]]
    """
    out = [None] * len(orig_name_to_ind)
    for name, i in orig_name_to_ind.items():
        univ_names = mapping[name]
        out[i] = [univ_name_to_ind[c] for c in univ_names] if univ_names != ['ignore'] else []
    assert not any(x is None for x in out)
    return out


def mapping_encode(mapping: T.Dict[str, T.Iterable[str]], dataset_c2i: T.Dict[str, int],
                   universal_c2i: T.Dict[str, int]) -> T.Dict[int, T.Iterable[int]]:  # JŠ
    """Replaces class names with class indices in a 1-to-many class mapping.

    Args:
        mapping: A lists of lists containing elementary class names.
            `mapping[k]` contains names of all elementary classes that a
            superclass with name `k` comprises.
        dataset_c2i: A mapping from superclass names to indices.
        universal_c2i: A mapping from elementary class names to indices.

    Returns:
        Dict[int, Iterable[int]]
    """
    dataset_i2c = {v: k for k, v in dataset_c2i.items()}
    max_index = max(dataset_c2i.values())
    m_list = [None] * (max_index + 1)
    for i in range(max_index + 1):
        class_name = dataset_i2c[i]
        univ_names = mapping[class_name]
        univ_indices = [] if univ_names == ['ignore'] else [universal_c2i[n] for n in univ_names]
        m_list[i] = univ_indices
    return m_list


def class_mapping_matrix(mapping: T.Sequence[T.Iterable[int]], univ_class_count, dtype=torch.int16):
    """Creates a matrix such that `mapping_matrix[k, c] = (c in mapping[k])`.

    Args:
        mapping: A lists of iterables containing elementary class indices.
            `mapping[i]` contains indices of all elementary classes that a
            superclass with index `i` comprises.
        univ_class_count: number of universal/elementary classes.
        dtype: Tensor data type.

    Returns:
        mapping_matrix (Tensor): A matrix representing mapping from superclasses to
            elementary classes. The shape is (K,C), where `C` is the number of
            elementary classes, and `K=len(mapping)<C` the number of superclasses.
    """
    cmm = torch.zeros((len(mapping), univ_class_count), dtype=dtype)
    for k, classes in enumerate(mapping):
        for c in classes:
            cmm[k, c] = 1
    return cmm


def superclass_mask(label: torch.Tensor, class_mapping: T.Dict[int, int], classes_last=False,
                    batch=True):
    """

    Args:
        label: An integer-valued tensor with shape (N,H,W).
        class_mapping: A matrix representing mapping from superclasses to
            elementary classes. The shape is (K,C), where C is the number of
            elementary classes, and K<C the number of superclasses.
        classes_last (bool): Whether the output tensor shape should be (N,K,H,W)
            (when `False`, default) or (N,H,W,K) (when `True`).
        batch (bool): Whether the input is a (N,H,W)-shaped batch (default) or
            a single (H,W)-shaped instance (when `False`). If `False`, the
            output shape will have no "batch" dimension.

    Returns:
         A tensor with shape (N,H,W,K).
    """
    if not batch:
        return _superclass_mask_single(label, class_mapping)
    target_flat = label.view(-1, 1)
    index = target_flat.expand(target_flat.shape[0], class_mapping.shape[1])
    mask = torch.gather(input=class_mapping, dim=0, index=index) \
        .view(*label.shape, class_mapping.shape[1])
    return mask if classes_last else mask.transpose(1, -1)


def _superclass_mask_single(target: torch.Tensor, class_mapping, classes_last=False):
    return superclass_mask(target.unsqueeze(0), class_mapping, classes_last=classes_last).squeeze(0)


def superclass_probs(probs: torch.Tensor, class_mapping: torch.Tensor):
    """ Transforms a C-class probabilities tensor with shape (N,C,...) into
    a K-class probabilities tensor with shape (N,K,...) according to a class
    mapping matrix.

    Args:
        probs: A C-class probabilities tensor with shape (N,C,...).
        class_mapping: A {0, 1}-valued K×C matrix generally representing a
            1-to-many mapping of classes, i.e. `class_mapping[k, c]` indicates
            whether `k` is the index of an output class that is a superclass of
            the input class with index `c`.

    Returns:
        Tensor: A K-class probabilities tensor with shape (N,K,...).
    """
    return torch.einsum("nc..., kc -> nk...", probs, class_mapping)

# target = torch.randint(high=3, size=(1, 4, 2))
# mapping_matrix = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=torch.uint8)
# mask = superclass_mask(target, mapping_matrix)

# def superclass_mask_naive(target: torch.Tensor, class_mapping, classes_last=False):
#     K, C = class_mapping.shape
#     mask = torch.zeros((*target.shape, C), dtype=class_mapping.dtype)
#     for n in range(mask.shape[0]):
#         for i in range(mask.shape[1]):
#             for j in range(mask.shape[2]):
#                 mask[n, i, j, :] = class_mapping[target[n, i, j]]
#     return mask if classes_last else mask.transpose(1, -1)

# from vidlu.utils import tree
#
#
# class IndexTree(dict):
#     def __init__(self, item, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.item = item  # flat class index
#         if len(self) > 0:
#             start = next(iter(self.values())).item
#             assert all(i == c.item for i, c in enumerate(self.values(), start=start))
#
#     def get(self, path):
#         if not isinstance(path, str):
#             if isinstance(path[0], int):
#                 return tree.deep_index_get(self, path).item
#             else:
#                 return tree.deep_get(self, path).item
#         return super().get(path)
#
#     def find_path(self, item):
#         path = []
#         tree = self
#         while True:
#             found = False
#             for k, v in reversed(tree.items()):
#                 if v.item <= item:
#                     found = True
#                     break
#             if not found:
#                 raise ValueError(item)
#             path.append(k)
#             tree = tree[k]
#             # if no child has the same item
#             if tree.item == item and (len(tree) == 0 or next(tree.values()).item != item):
#                 break
#         return tuple(path)
#
#
# def encode_tree_prefix(class_tree):
#     return {i: encode_tree_prefix(v) if isinstance(v, dict) else v
#             for i, v in enumerate(class_tree.values())}
#
#
# def encode_tree(class_tree, first_child_index=0, path=()):
#     children = dict()
#     child_indices = list(range(first_child_index, first_child_index + len(class_tree)))
#     next_index = child_indices[-1] + 1
#     for ci, (k, v) in zip(child_indices, class_tree.items()):
#         child_path = (*path, k)
#         if isinstance(v, dict):
#             children[k], next_index = encode_tree_flat(v, next_index, child_path)
#             children[k].item = ci
#         else:
#             children[k] = IndexTree(ci)
#     return IndexTree(first_child_index, children), next_index
#
#
# def encode_tree_flat(class_tree, only_leaves, path=(), this_index=None, first_child_index=0):
#     children = dict()
#     child_indices = list(range(first_child_index, first_child_index + len(class_tree)))
#     next_index = child_indices[-1] + 1
#     for ci, (k, v) in zip(child_indices, class_tree.items()):
#         child_path = (*path, k)
#         if isinstance(v, dict):
#             children[k], next_index = encode_tree_flat(v, only_leaves, child_path, next_index)
#             children[k].item = ci
#         else:
#             children[k] = IndexTree(ci)
#     return IndexTree(first_child_index, children), next_index + 1 - int(only_leaves)
#
#
# def encode_tree_softmax(class_tree):
#     return {i: encode_tree(v) if isinstance(v, dict) else v
#             for i, v in enumerate(class_tree.values())}
