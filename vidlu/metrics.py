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

    def update(self, iter_output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


def _get_iter_output(iter_output, name):
    path = name.split('.')
    for part in path:
        iter_output = iter_output[part]
    return iter_output


def multiclass_confusion_matrix(true, pred, class_count, dtype=None, batch=False):
    """Computes a multi-class confusion matrix.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): a vector of integers representing predicted classes.
        class_count (int): number of classes.
        dtype (optional): confusion matrix data type.

    Returns:
        A confusion matrix with shape (class_count, class_count).
    """
    return soft_pred_multiclass_confusion_matrix(true, one_hot(pred, class_count, dtype=torch.float64),
                                                 batch=batch) \
        .to(dtype or torch.int64)


def soft_pred_multiclass_confusion_matrix(true, pred, dtype=None, batch=False):
    """Computes a soft multi-class confusion matrix from probabilities.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): an array consisting of vectors representing predicted class
            probabilities.
        dtype (optional): confusion matrix data type.

    Returns:
        A soft confusion matrix with shape (class_count, class_count)
    """
    dtype = dtype or pred.dtype
    class_count = pred.shape[-1]
    non_ignored = true != -1
    if batch:
        assert torch.all(non_ignored)
    return all_soft_multiclass_confusion_matrix(one_hot(true[non_ignored], class_count, dtype=dtype),
                                                pred[non_ignored].to(dtype), batch=batch)

    # 3 - 4 times faster than
    # cm = torch.empty(list(true.shape[:int(batch)]) + [class_count] * 2,
    #                  dtype=dtype or torch.float64, device=true.device)
    # if batch:
    #     for c in range(class_count):
    #         cm[:, c, :] = pred[:, true == c, :].sum(int(batch))
    # else:
    #     for c in range(class_count):
    #         cm[c, :] = pred[true == c, :].sum(int(batch))
    # return cm


def all_soft_multiclass_confusion_matrix(true, pred, dtype=None, batch=False):
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
    return torch.einsum("bni,bnj->bij" if batch else "ni,nj->ij", true, pred)


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
    def __init__(self, class_count, target_name="target",
                 hard_prediction_name="other_outputs.hard_prediction",
                 metrics=('A', 'mP', 'mR', 'mIoU'), device=None):
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.int64, requires_grad=False,
                              device=device)
        self.target_name = target_name
        self.hard_prediction_name = hard_prediction_name
        self.metrics = metrics

    @torch.no_grad()
    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_output):
        true = _get_iter_output(iter_output, self.target_name).flatten()
        pred = _get_iter_output(iter_output, self.hard_prediction_name).flatten()
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
    def update(self, iter_output):
        self._sum += self.value_extractor(iter_output)
        self._n += 1

    def compute(self):
        return {self.name: self._sum.value / self._n}


class HarmonicMeanMetric(_MeanMetric):
    def update(self, iter_output):
        self._sum += 1 / self.value_extractor(iter_output)
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
    def update(self, iter_output):
        val = self.extract_func(iter_output)
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
    def update(self, iter_output):
        if self.metrics is None:
            self.metrics = [self.metric_f(k) for k, v in iter_output.items() if self.filter(k, v)]
        for m in self.metrics:
            m.update(iter_output)

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
    def __init__(self, class_count, target_name="target", probs_name="other_outputs.probs",
                 metrics=('A', 'mP', 'mR', 'mIoU')):
        super().__init__()
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.float64, requires_grad=False)
        self.target_name = target_name
        self.probs_name = probs_name
        self.metrics = metrics

    @torch.no_grad()
    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_output):
        true = _get_iter_output(iter_output, self.target_name).flatten()
        pred = _get_iter_output(iter_output, self.probs_name).permute(0, 2, 3, 1)
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
