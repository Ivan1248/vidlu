import math
import typing as T
from collections import defaultdict

import torch

import vidlu.modules.elements as E
from vidlu.ops import one_hot


# Dict mappings


def invert_many_to_one_mapping(mapping: T.Mapping):
    result = defaultdict(list)
    for k, v in mapping.items():
        result[v].append(k)
    return result


def invert_one_to_many_mapping(mapping: T.Mapping):
    return {v: k for k, vs in mapping.items() for v in vs}


# Class names and indices

def _to_inverted_index_if_list(value_to_ind):
    return ({v: i for i, v in enumerate(value_to_ind)}
            if isinstance(value_to_ind, (T.Sequence, T.KeysView, T.ValuesView)) else
            value_to_ind)


InvertedIndex = T.Dict[str, int]


def encode_one_to_many_mapping(mapping: T.Dict[str, T.Iterable[str]],
                               value_to_ind: T.Union[InvertedIndex, T.Sequence[str]],
                               key_to_ind: T.Union[InvertedIndex, T.Sequence[str]] = None,
                               return_list=False) -> T.Union[
    T.Dict[int, T.List[int]], T.List[T.List[int]]]:
    """Replaces class names with class indices in a 1-to-many class mapping.
    Args:
        mapping: A mapping from target (superclass) names to elementary class names.
        value_to_ind: A mapping from target class names to indices.
        key_to_ind: A mapping from superclass names to indices.
        return_list: Whether to return a list instead of Dict[int, List[int]].
    Returns:
        Union[Dict[int, List[int]], List[List[int]]]
    """
    value_to_ind = _to_inverted_index_if_list(value_to_ind)
    key_to_ind = _to_inverted_index_if_list(key_to_ind or mapping.keys())

    out = {key_to_ind[k]: [value_to_ind[c] for c in subs] for k, subs in mapping.items()}
    return [out[i] for i in range(len(key_to_ind))] if return_list else out


def encode_many_to_one_mapping(mapping: T.Dict[str, str],
                               value_to_ind: T.Union[InvertedIndex, T.Sequence[str]],
                               key_to_ind: T.Union[InvertedIndex, T.Sequence[str]] = None,
                               return_list=False) -> T.Union[
    T.Dict[int, T.List[int]], T.List[T.List[int]]]:
    """Replaces class names with class indices in a 1-to-many class mapping.
    Args:
        mapping: A mapping from elementary names to target class (superclass) names.
        value_to_ind: A mapping from target class names to indices.
        key_to_ind: A mapping from elementary names to indices.
        return_list: Whether to return a list instead of Dict[int, int].
    Returns:
        Union[Dict[int, int], List[int]]
    """
    value_to_ind = _to_inverted_index_if_list(value_to_ind)
    key_to_ind = _to_inverted_index_if_list(key_to_ind or mapping.keys())

    out = {key_to_ind[k]: value_to_ind[v] for k, v in mapping.items()}
    return [out[i] for i in range(len(key_to_ind))] if return_list else out


# Class indices

def class_mapping_matrix(mapping: T.Sequence[T.Iterable[int]], elem_class_count, dtype=torch.bool):
    """Creates a matrix such that `mapping_matrix[k, c] = (c in mapping[k])`.

    Args:
        mapping: A lists of iterables containing elementary class indices.
            `mapping[i]` contains indices of all elementary classes that a
            superclass with index `i` comprises.
        elem_class_count: number of universal/elementary classes.
        dtype: Tensor data type.

    Returns:
        mapping_matrix (Tensor): A matrix representing mapping from superclasses to
            elementary classes. The shape is (K,C), where `C` is the number of
            elementary classes, and `K=len(mapping)<C` the number of superclasses.
    """
    cmm = torch.zeros((len(mapping), elem_class_count), dtype=dtype)
    for k, classes in enumerate(mapping):
        for c in classes:
            cmm[k, c] = 1
    return cmm


def normalize_mapping(mapping: T.Dict[object, int], ignore_key: int = None,
                      ignore_index: int = -1) -> T.Dict[object, int]:
    """Normalizes an ordered mapping from keys to indices so that indices of
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


def superclass_mask(label: torch.Tensor, class_mapping: torch.Tensor, classes_last=False,
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
    target_flat = label.view(-1, 1)  # NHW,1
    index = target_flat.expand(target_flat.shape[0], class_mapping.shape[1])  # NHW,K
    mask = torch.gather(class_mapping, dim=0, index=index).view(*label.shape,
                                                                class_mapping.shape[1])
    return mask.permute(0, -1, *list(range(1, len(mask.shape) - 1))) if classes_last else mask


def _superclass_mask_single(label: torch.Tensor, class_mapping: torch.Tensor):
    return class_mapping[label.view(-1)].view(*label.shape, -1).permute(2, 0, 1)
    # return superclass_mask(target.unsqueeze(0), class_mapping,
    # classes_last=classes_last).squeeze(0)


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
    N, _, H, W = probs.shape
    K, C = class_mapping.shape
    return torch.bmm(class_mapping.view(1, K, C).repeat(N, 1, 1), probs.view(N, C, -1)).view(N, K,
                                                                                             H, W)


def filter_mapping_eval(class_mapping_matrix, class_to_eval, class_to_index, void_mapping):
    eval_indices = []
    for cname in class_to_eval:
        idx, is_eval = class_to_index[cname], class_to_eval[cname]
        if is_eval:
            eval_indices += [idx]
    eval_selection = torch.tensor(sorted(eval_indices)).long()
    if void_mapping is False:
        return class_mapping_matrix[eval_selection]
    ignore_vector = class_mapping_matrix.sum(0).logical_not()
    return torch.cat([class_mapping_matrix[eval_selection], ignore_vector.unsqueeze(0)])


def create_class_mappings(elem_class_to_ind, class_to_ind, dataset_c2e, name_mapping,
                          void_mapping=False):
    encoded_mapping = encode_one_to_many_mapping(name_mapping, elem_class_to_ind, class_to_ind)
    class_mapping_train = class_mapping_matrix(encoded_mapping,
                                               len(
                                                   elem_class_to_ind.keys()) - 1)
    class_mapping_eval = filter_mapping_eval(class_mapping_train, dataset_c2e, class_to_ind,
                                             void_mapping)
    return class_mapping_train, class_mapping_eval


class SoftClassMapping(E.Module):
    def __init__(self, target_class_count, elem_class_count, include_other=False,
                 init=torch.nn.init.normal_):
        super().__init__()
        self.include_other = include_other
        self.weights = torch.nn.Parameter(
            torch.zeros(target_class_count + int(include_other), elem_class_count),
            requires_grad=True)
        with torch.no_grad():
            init(self.weights)

    def forward(self, univ_probs):
        N, C, *HW = univ_probs.shape  # C
        mapping_mat = self.weights.mul(1).softmax(0)  # K×C
        if self.include_other:
            mapping_mat = mapping_mat[:-1]
        K, C_ = mapping_mat.shape
        result = mapping_mat.unsqueeze(0).expand(N, K, C).bmm(univ_probs.view(N, C, -1)) \
            .view(N, K, *HW)
        return result


class SoftClassMappingTargetProb(E.Module):
    def __init__(self, soft_class_mapping):
        self.soft_class_mapping = soft_class_mapping

    def forward(self, univ_probs, target, class_count):
        target_oh = one_hot(target, class_count)


class MultiSoftClassMapping(E.ModuleTable):
    def __init__(self, name_to_target_class_count, elem_class_count=None, include_other=False,
                 init=None):
        super().__init__()
        self.name_to_target_class_count = name_to_target_class_count
        self.elem_class_count = elem_class_count
        self.include_other = include_other
        self.init = getattr(SoftMappingInit, init) if isinstance(init, str) else init
        if elem_class_count is not None:
            self._build()
            self._built = True

    def _build(self):
        self.add(**{name: SoftClassMapping(C, self.elem_class_count,
                                           include_other=self.include_other)
                    for name, C in self.name_to_target_class_count.items()})
        if self.init is not None:
            self.init(self)

    def build(self, univ_probs, names):
        if self.elem_class_count is None:
            self.elem_class_count = univ_probs.shape[1]
            self._build()
            self.to(device=univ_probs.device, dtype=univ_probs.dtype)

    def forward(self, univ_probs, names):
        return [self[name](up.unsqueeze(0))[0] for up, name in zip(univ_probs, names)]


def get_index_offsets(class_counts):
    """The last element is the total number of indices."""
    offsets = [0]
    for k in class_counts:
        offsets = offsets[-1] + k
    return offsets


class MultiSoftClassMappingEfficient(E.ModuleTable):
    def __init__(self, name_to_target_class_count, elem_class_count):
        super().__init__()
        self.name_to_ind = {name: i for i, name in enumerate(name_to_target_class_count)}
        self.offsets = get_index_offsets(name_to_target_class_count.values())
        self.weights = torch.zeros(self.offsets[-1], elem_class_count, requires_grad=True)

    def forward(self, univ_probs, names):
        indices = []
        for name in names:
            i = self.name_to_ind[name]
            indices.append([range(self.offsets[i], self.offsets[i + 1])])
        N, C, *HW = univ_probs.shape
        mapping_mat = self.weights.softmax(1)
        KC = mapping_mat.shape
        return mapping_mat.unsqueeze(0).expand(N, *KC).bmm(univ_probs.view(N, C, -1))


class SoftMappingInit:
    @staticmethod
    @torch.no_grad()
    def one_to_one(module: MultiSoftClassMapping, p=None, logit=None):
        if (logit is None) == (p is None):
            raise ValueError("Either prob or logit should be provided.")

        offset = 0
        for m in module.children():
            w = m.weights
            C = w.shape[0] - int(m.include_other)
            if p is not None:
                logit = math.log(C * p / (1 - p))
            w.zero_()
            for i in range(C):
                w[i, offset + i] = logit
            if m.include_other:
                w[-1, :offset] = logit
                w[-1, offset + C:] = logit
            offset += C

    @staticmethod
    @torch.no_grad()
    def one_to_one_same(module: MultiSoftClassMapping, p=None, logit=None):
        if (logit is None) == (p is None):
            raise ValueError("Either p (probability) or logit should be provided.")

        for m in module.children():
            w = m.weights
            C = w.shape[0] - int(m.include_other)
            if p is not None:
                logit = math.log(C * p / (1 - p))
            w.zero_()
            for i in range(C):
                w[i, i] = logit
            if m.include_other:
                w[-1, C:] = logit

    @staticmethod
    @torch.no_grad()
    def uniform(module: MultiSoftClassMapping, p_other=None, logit_other=None):
        if (logit_other is None) == (p_other is None):
            raise ValueError("Either p_other or logit_other should be provided.")

        for m in module.children():
            w = m.weights
            C = w.shape[0] - int(m.include_other)
            if p_other is not None:
                logit_other = math.log(C * p_other / (1 - p_other))
            w.zero_()
            if m.include_other:
                w[-1, :] = logit_other
