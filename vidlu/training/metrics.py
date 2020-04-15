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


@torch.no_grad()
def multiclass_confusion_matrix(true, pred, class_count):
    """ Computes a multiclass confusion matrix.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): a vector of integers representing predicte classes.
        class_count (int):

    Returns:
        A confusion matrix with shape (class_count, class_count).
    """
    pred = one_hot(pred, class_count, dtype=torch.int64)
    cm = torch.zeros([class_count] * 2, requires_grad=False,
                     dtype=torch.int64, device='cpu')  # TODO: why CPU
    for c in range(class_count):
        cm[c, :] = pred[true == c, :].sum(0)
    return cm


@torch.no_grad()
def soft_multiclass_confusion_matrix(true, pred, class_count):
    """ Computes a soft multiclass confusion matrix from probabilities.

    Args:
        true (Tensor): a vector of integers representing true classes.
        pred (Tensor): an array consisting of vectors representing predicted class
            probabilities.
        class_count (int):

    Returns:
        A soft confusion matrix with shape (class_count, class_count)
    """
    cm = torch.zeros([class_count] * 2, requires_grad=False, dtype=torch.float64, device='cpu')
    for c in range(class_count):
        cm[c, :] = pred[true == c, :].sum(0)
    return cm


@torch.no_grad()
def compute_classification_metrics(cm, returns=('A', 'mP', 'mR', 'mF1', 'mIoU'), eps=1e-8):
    """ Computes macro-averaged classification evaluation metrics based on the
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
    locs = locals()
    return dict((x, locs[x]) for x in returns)


class ClassificationMetrics(AccumulatingMetric):
    def __init__(self, class_count, target_name="target",
                 hard_prediction_name="other_outputs.hard_prediction",
                 metrics=('A', 'mP', 'mR', 'mIoU')):
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.int64, requires_grad=False)
        self.target_name = target_name
        self.hard_prediction_name = hard_prediction_name
        self.metrics = metrics

    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_output):
        true = _get_iter_output(iter_output, self.target_name).flatten()
        pred = _get_iter_output(iter_output, self.hard_prediction_name).flatten()
        self.cm += multiclass_confusion_matrix(true, pred, self.class_count)

    def compute(self, eps=1e-8):
        return compute_classification_metrics(self.cm.cpu().numpy(), returns=self.metrics, eps=eps)


class _MeanMetric(AccumulatingMetric, metaclass=ABCMeta):
    def __init__(self, name, extract_func=None):
        self.name = name
        self.extract_func = extract_func or (lambda x: x[name])
        self.reset()

    def reset(self):
        self._sum = KleinSum()
        self._n = EPS


class AverageMetric(_MeanMetric):
    def update(self, iter_output):
        self._sum += self.extract_func(iter_output)
        self._n += 1

    def compute(self):
        return {self.name: self._sum.value / self._n}


class HarmonicMeanMetric(_MeanMetric):
    def update(self, iter_output):
        self._sum += 1 / self.extract_func(iter_output)
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

    def update(self, iter_output):
        val = self.extract_func(iter_output)
        self._ext = self.extremum_func(self._ext or val, val)

    def compute(self):
        return {self.name: self._ext}


class MaxMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, max, extract_func=extract_func)


class MinMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, min, extract_func=extract_func)


class _MultiMetric(AccumulatingMetric):
    def __init__(self, name_filter, metric_f):
        self.name_filter = name_filter
        self.metrics = []
        self.metric_f = metric_f

    def reset(self):
        self.metrics = []

    @torch.no_grad()
    def update(self, iter_output):
        if len(self.metrics) == 0:
            self.metrics = [self.metric_f(k) for k in iter_output.keys() if self.name_filter(k)]
        for m in self.metrics:
            m.update(iter_output)

    def compute(self):
        result = dict()
        for m in self.metrics or ():
            result.update(m.compute())
        return result

class AverageMultiMetric(_MultiMetric):
    def __init__(self, name_filter):
        super().__init__(name_filter=name_filter, metric_f=AverageMetric)


class HarmonicMeanMultiMetric(_MultiMetric):
    def __init__(self, name_filter):
        super().__init__(name_filter=name_filter, metric_f=HarmonicMeanMetric)


class MaxMultiMetric(_MultiMetric):
    def __init__(self, name_filter):
        super().__init__(name_filter=name_filter, metric_f=MaxMetric)

class MinMultiMetric(_MultiMetric):
    def __init__(self, name_filter):
        super().__init__(name_filter=name_filter, metric_f=MinMetric)


class SoftClassificationMetrics(AccumulatingMetric):
    def __init__(self, class_count, target_name="target", probs_name="other_outputs.probs",
                 metrics=('A', 'mP', 'mR', 'mIoU')):
        super().__init__()
        self.class_count = class_count
        self.cm = torch.zeros([class_count] * 2, dtype=torch.float64, requires_grad=False)
        self.target_name = target_name
        self.probs_name = probs_name
        self.metrics = metrics

    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_output):
        true = _get_iter_output(iter_output, self.target_name).flatten()
        pred = _get_iter_output(iter_output, self.probs_name).permute(0, 2, 3, 1)
        pred = pred.flatten().view(-1, pred.shape[-1])
        self.cm += soft_multiclass_confusion_matrix(true, pred, self.class_count)

    compute = ClassificationMetrics.compute


def with_suffix(metric_class, suffix):
    class MetricReturnsSuffixWrapper(metric_class):
        @wraps(metric_class.compute)
        def compute(self, *args, **kwargs):
            return {f"{k}_{suffix}": v for k, v in super().compute(*args, **kwargs).items()}

        def __getattr__(self, item):
            return getattr(self.metric, item)

    return MetricReturnsSuffixWrapper
