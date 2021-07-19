import functools
from abc import ABCMeta

import numpy as np
import torch
from functools import wraps

from vidlu.ops import one_hot
from vidlu.utils.num import KleinSum

EPS = 1e-8


# from dlu #########################################################################################

class AccumulatingMetric:
    def reset(self):
        raise NotImplementedError()

    def update(self, iter_result):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


def multiclass_confusion_matrix(true, pred, class_count, dtype=None, batch=False,
                                use_bincount=False):
    """Computes a multi-class confusion matrix.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): a vector of integers representing predicted classes.
        class_count (int): number of classes.
        dtype (optional): confusion matrix data type.

    Returns:
        A confusion matrix with shape (class_count, class_count).
    """
    if use_bincount:
        non_ignored = true != -1
        indices = class_count * true[non_ignored] + pred[non_ignored]
        cm = torch.bincount(indices, minlength=class_count ** 2).reshape(class_count, class_count)
    else:
        cm = soft_pred_multiclass_confusion_matrix(
            true, one_hot(pred, class_count, dtype=torch.float64), batch=batch)
    return cm.to(dtype or torch.int64)


def soft_pred_multiclass_confusion_matrix(true, pred, dtype=None, batch=False, loop_version=None):
    """Computes a soft multi-class confusion matrix from probabilities.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): an array consisting of vectors representing predicted class
            probabilities.
        dtype (optional): confusion matrix data type.

    Returns:
        A soft confusion matrix with shape (class_count, class_count)
    """
    if loop_version is None:
        if true.device.type == 'cuda':
            loop_version = "3090" in torch.cuda.get_device_name(true.device.index)
    if not loop_version:
        dtype = dtype or pred.dtype
        class_count = pred.shape[-1]
        non_ignored = true != -1
        if batch:
            assert torch.all(non_ignored)
        return all_soft_multiclass_confusion_matrix(
            one_hot(true[non_ignored], class_count, dtype=dtype),
            pred[non_ignored].to(dtype), batch=batch)
    else:  # usually 3 - 4 times slower
        class_count = pred.shape[-1]
        cm = torch.empty(list(true.shape[:int(batch)]) + [class_count] * 2,
                         dtype=dtype or torch.float64, device=true.device)
        if batch:
            for c in range(class_count):
                cm[:, c, :] = pred[:, true == c, :].sum(int(batch))
        else:
            for c in range(class_count):
                cm[c, :] = pred[true == c, :].sum(int(batch))
        return cm


def all_soft_multiclass_confusion_matrix(true: torch.Tensor, pred: torch.Tensor, dtype=None,
                                         batch=False):
    """Computes a soft multi-class confusion matrix from probabilities.

    Args:
        true (Tensor): an array consisting of vectors representing true class
            probabilities.
        pred (Tensor): an array consisting of vectors representing predicted class
            probabilities.
        dtype (optional): confusion matrix data type.

    Returns:
        A soft confusion matrix with shape (class_count, class_count)
    """
    if dtype is not None:
        true, pred = true.to(dtype), pred.to(dtype)
    # return true.new_ones((true.shape[-1], true.shape[-1]))
    return torch.einsum("bni,bnj->bij" if batch else "ni,nj->ij", true, pred)
    return torch.bmm(true.permute(1, 0).unsqueeze(0).to(dtype=torch.float16),
                     pred.unsqueeze(0).to(dtype=torch.float16)).squeeze(0)
    # return torch.einsum("ni,nj->ij", true, pred)


def classification_metrics_np(cm, returns=('A', 'mP', 'mR', 'mF1', 'mIoU'), eps=1e-8):
    """Computes macro-averaged classification evaluation metrics based on the
        accumulated confusion matrix and clears the confusion matrix.

    Args:
        cm (np.array): a confusion matrix.
        returns (Sequence): a list of metrics that should be returned.
        eps (float): a number to add to the denominator to avoid division by 0.

    Returns:
        A dictionary with computed classification evaluation metrics.
    """
    tp = np.diag(cm)
    actual_pos = cm.sum(axis=1)
    pos = cm.sum(axis=0) + eps
    fp = pos - tp

    with np.errstate(divide='ignore', invalid='ignore'):
        P = tp / pos
        R = tp / actual_pos
        F1 = 2 * P * R / (P + R)
        IoU = tp / (actual_pos + fp)
    P, R, F1, IoU = map(np.nan_to_num, [P, R, F1, IoU])  # 0 where tp=0
    mP, mR, mF1, mIoU = map(np.mean, [P, R, F1, IoU])
    A = tp.sum() / pos.sum()
    locals_ = locals()
    if isinstance(returns, str):
        return locals_[returns]
    return {k: locals_[k] for k in returns}


def classification_metrics(cm, returns=('A', 'mP', 'mR', 'mF1', 'mIoU', 'cm'), eps=1e-8):
    """Computes macro-averaged classification evaluation metrics based on the
    accumulated confusion matrix.

    Supports batches (when `cm` is a batch of matrices).

    Args:
        cm (np.array): a confusion matrix.
        returns (Sequence): a list of metrics that should be returned.
        eps (float): a number to add to the denominator to avoid division by 0.

    Returns:
        A dictionary with computed classification evaluation metrics.
    """
    is_batch = int(cm.dim() == 3)
    tp = cm.diagonal(dim1=is_batch, dim2=is_batch + 1)
    actual_pos = cm.sum(dim=is_batch + 1)
    pos = cm.sum(dim=is_batch)
    fp = pos - tp

    tp = tp.float()
    P = tp / pos
    R = tp / actual_pos
    F1 = 2 * P * R / (P + R)
    IoU = tp / (actual_pos + fp)
    for x in [P, R, F1, IoU]:  # 0 where tp=0
        x[torch.isnan(x)] = 0

    mP, mR, mF1, mIoU = map(torch.mean, [P, R, F1, IoU])
    A = tp.sum(dim=is_batch) / pos.sum(dim=is_batch)
    locals_ = locals()
    if isinstance(returns, str):
        return locals_[returns]
    return {k: locals_[k] for k in returns}


def mIoU(cm, eps=1e-8):
    is_batch = int(cm.dim() == 3)
    tp = cm.diagonal(dim1=is_batch, dim2=is_batch + 1)
    actual_pos = cm.sum(dim=is_batch + 1)
    pos = cm.sum(dim=is_batch)
    fp = pos - tp
    tp = tp.float()
    IoU = tp / (actual_pos + fp)
    IoU[torch.isnan(IoU)] = 0
    return IoU.mean()


class ClassificationMetrics(AccumulatingMetric):
    def __init__(self, class_count, get_target=lambda r: r.target,
                 get_hard_prediction=lambda r: r.out.argmax(1),
                 metrics=('A', 'mP', 'mR', 'mIoU'), device=None):
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.int64, requires_grad=False,
                              device=device)
        self.get_target = get_target
        self.get_hard_prediction = get_hard_prediction
        self.metrics = metrics

    @torch.no_grad()
    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_result):
        true = self.get_target(iter_result).flatten()
        pred = self.get_hard_prediction(iter_result).flatten()
        cm = multiclass_confusion_matrix(true, pred, self.class_count)
        if self.cm.device != cm.device:
            self.cm = self.cm.to(cm.device)
        self.cm += cm

    @torch.no_grad()
    def compute(self, eps=1e-8):
        return {k: v.item() if v.dim() == 0 else v.cpu().numpy().copy() for k, v in
                classification_metrics(self.cm, returns=self.metrics, eps=eps).items()}


class _MeanMetric(AccumulatingMetric, metaclass=ABCMeta):
    def __init__(self, name, value_extractor=None):
        self.name = name
        self.value_extractor = value_extractor or (lambda x: x[name])
        self.reset()

    def reset(self):
        self._sum = KleinSum()
        self._n = EPS


class AverageMetric(_MeanMetric):
    def update(self, iter_result):
        self._sum += self.value_extractor(iter_result)
        self._n += 1

    def compute(self):
        return {self.name: self._sum.value / self._n}


class HarmonicMeanMetric(_MeanMetric):
    def update(self, iter_result):
        self._sum += 1 / self.value_extractor(iter_result)
        self._n += 1

    def compute(self):
        return {self.name: self._n / self._sum.value}


class _ExtremumMetric(AccumulatingMetric):
    def __init__(self, name, extremum_func, extract_func=None):
        self.name = name
        self.extremum_func = extremum_func
        self.extract_func = extract_func or (lambda x: x[name])
        self.reset()

    def reset(self):
        self._ext = None

    @torch.no_grad()
    def update(self, iter_result):
        val = self.extract_func(iter_result)
        self._ext = self.extremum_func(self._ext or val, val)

    @torch.no_grad()
    def compute(self):
        return {self.name: self._ext}


class MaxMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, max, extract_func=extract_func)


class MinMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, min, extract_func=extract_func)


class _MultiMetric(AccumulatingMetric):
    def __init__(self, filter, metric_f):
        self.filter = filter
        self.metrics = None
        self.metric_f = metric_f

    def reset(self):
        self.metrics = None

    @torch.no_grad()
    def update(self, iter_result):
        if self.metrics is None:
            self.metrics = [self.metric_f(k) for k, v in iter_result.items() if self.filter(k, v)]
        for m in self.metrics:
            m.update(iter_result)

    @torch.no_grad()
    def compute(self):
        result = dict()
        for m in self.metrics or ():
            result.update(m.compute())
        return result


class AverageMultiMetric(_MultiMetric):
    def __init__(self, filter):
        super().__init__(filter=filter, metric_f=AverageMetric)


class HarmonicMeanMultiMetric(_MultiMetric):
    def __init__(self, filter):
        super().__init__(filter=filter, metric_f=HarmonicMeanMetric)


class MaxMultiMetric(_MultiMetric):
    def __init__(self, filter):
        super().__init__(filter=filter, metric_f=MaxMetric)


class MinMultiMetric(_MultiMetric):
    def __init__(self, filter):
        super().__init__(filter=filter, metric_f=MinMetric)


class SoftClassificationMetrics(AccumulatingMetric):
    def __init__(self, class_count, get_target=lambda r: r.target,
                 get_probs=lambda r: r.out.softmax(1),
                 metrics=('A', 'mP', 'mR', 'mIoU')):
        super().__init__()
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.float64, requires_grad=False)
        self.get_target = get_target
        self.get_probs = get_probs
        self.metrics = metrics

    @torch.no_grad()
    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_result):
        true = self.get_target(iter_result).flatten()
        pred = self.get_probs(iter_result).permute(0, 2, 3, 1)
        pred = pred.flatten().view(-1, pred.shape[-1])
        self.cm += soft_pred_multiclass_confusion_matrix(true, pred, self.class_count)

    compute = ClassificationMetrics.compute


def with_suffix(metric_class, suffix):
    class MetricReturnsSuffixWrapper(metric_class):
        @wraps(metric_class.compute)
        def compute(self, *args, **kwargs):
            return {f"{k}_{suffix}": v for k, v in super().compute(*args, **kwargs).items()}

        def __getattr__(self, item):
            return getattr(self.metric, item)

    return MetricReturnsSuffixWrapper
