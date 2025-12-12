import functools
from abc import ABCMeta
import typing as T

import numpy as np
import torch
from functools import wraps

from vidlu.ops import one_hot
from vidlu.utils.num import KleinSum
from vidlu.torch_utils import retry_if_cuda_oom


EPS = 1e-8


# from dlu #########################################################################################

class AccumulatingMetric:
    """
    Abstract base class for metrics that can be computed in multiple iterations.
    """

    def reset(self):
        """Resets the state of the metric.
        """
        raise NotImplementedError()

    def update(self, iter_result: T.Mapping[str, T.Any]):
        """Updates the metric with results from a mini-batch.

        Args:
            iter_result (Mapping[str, Any]): A dictionary that contains the data necessary to
                compute the metrics.
        """
        raise NotImplementedError()

    def compute(self) -> T.Mapping[str, T.Any]:
        """Computes and returns the final metric values after all updates.

        Returns:
            Mapping[str, Any]: Mapping of metric names to values.
        """
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
    return torch.einsum("bni,bnj->bij" if batch else "ni,nj->ij", true, pred)
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


def masked_mean(values, mask):
    return (values if mask is None else values[mask]).mean()


def classification_metrics(cm, returns=('A', 'mP', 'mR', 'mF1', 'mIoU', 'cm'), eps=1e-8, only_present=False):
    """Computes macro-averaged classification evaluation metrics based on the
    accumulated confusion matrix.

    Supports batches (when `cm` is a batch of matrices).

    **Important**: By default, macro-averaged metrics (mP, mR, mF1, mIoU) are 
    computed over all classes, including classes with no samples (metric=0).
    
    This differs from sklearn's `f1_score(average='macro')`, which only averages
    over classes that appear in `y_true` or `y_pred`. Set `only_present=True` 
    to match sklearn's behavior.
    
    Args:
        cm (Tensor): A confusion matrix of shape (C, C) or batch (B, C, C).
        returns (Sequence): A list of metrics that should be returned.
        eps (float): A number to add to the denominator to avoid division by 0.
        only_present (bool): If True, compute macro metrics only over classes 
            with samples (matches sklearn). Default False (existing behavior).

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

    metrics = dict(P=P, R=R, F1=F1, IoU=IoU)

    # Compute macro metrics with optional masking
    mask = (actual_pos + fp) > 0 if only_present else None  
    metrics.update({'m' + k: masked_mean(v, mask) for k, v in metrics.items()})
    metrics['A'] = tp.sum(dim=is_batch) / pos.sum(dim=is_batch)
    metrics['num_correct'] = tp.sum(dim=is_batch)
    if isinstance(returns, str):
        return metrics[returns]
    return {k: metrics[k] for k in returns}


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
    """
    Computes classification metrics based on a confusion matrix.

    Supported metrics:
    - 'A': accuracy (Global accuracy: total TP / total samples)
    - 'mP': macro-averaged precision (mean of per-class precision)
    - 'mR': macro-averaged recall (mean of per-class recall)
    - 'mF1': macro-averaged F1 score (mean of per-class F1)
    - 'mIoU': macro-averaged intersection over union (mean of per-class IoU)
    - 'P': per-class precision (TP / (TP + FP))
    - 'R': per-class recall (TP / (TP + FN))
    - 'F1': per-class F1 score
    - 'IoU': per-class IoU
    - 'cm': confusion matrix
    
    **Important**: By default, macro-averaged metrics (mP, mR, mF1, mIoU) are 
    computed over all classes, including classes with no samples (metric=0).
    
    This differs from sklearn's `f1_score(average='macro')`, which only averages
    over classes that appear in `y_true` or `y_pred`. Set `only_present=True` 
    to match sklearn's behavior.
    """
    def __init__(self, class_count, get_target=lambda r: r.target,
                 get_hard_prediction=lambda r: r.out.argmax(1),
                 metrics=('A', 'mP', 'mR', 'mIoU'), device=None, cm=None,
                 only_present_classes=False):
        self.class_count = class_count
        if cm is not None:
            assert list(cm.shape) == [class_count] * 2
            self.cm = cm
        else:
            self.cm = torch.zeros([class_count] * 2, dtype=torch.int64, requires_grad=False,
                                  device=device)
        self.get_target = get_target
        self.get_hard_prediction = get_hard_prediction
        self.metrics = metrics
        self.only_present = only_present_classes

    @torch.no_grad()
    def reset(self):
        self.cm.fill_(0)

    @torch.no_grad()
    def update(self, iter_result):
        true = self.get_target(iter_result).flatten()
        pred = self.get_hard_prediction(iter_result).flatten()
        cm = retry_if_cuda_oom(multiclass_confusion_matrix)(true, pred, self.class_count)
        if self.cm.device != cm.device:
            self.cm = self.cm.to(cm.device)
        self.cm += cm

    @torch.no_grad()
    def compute(self, eps=1e-8):
        return {k: v.item() if v.dim() == 0 else v.cpu().numpy().copy() for k, v in
                classification_metrics(self.cm, returns=self.metrics, eps=eps, 
                                     only_present=self.only_present).items()}

    def __repr__(self):
        return f"{type(self).__name__}(class_count={self.class_count}, metrics={self.metrics})"


class ThresholdlessBinaryClassificationMetrics(AccumulatingMetric):
    def __init__(self, get_target=lambda r: r.target,
                 get_prediction=lambda r: r,
                 metrics=('AuROC', 'AuPR', 'FPR95')):
        self.reset()
        self.get_target = get_target
        self.get_prediction = get_prediction
        self.metrics = metrics

    @torch.no_grad()
    def reset(self):
        self.truths = []
        self.predictions = []

    @torch.no_grad()
    def update(self, iter_result):
        true = self.get_target(iter_result)
        non_ignored = true != -1
        true = true[non_ignored]
        pred = self.get_prediction(iter_result)[non_ignored]
        self.truths.extend(true.cpu().numpy().tolist())
        self.predictions.extend(pred.cpu().numpy().tolist())

    @torch.no_grad()
    def compute(self):
        from sklearn.metrics import average_precision_score, roc_curve, auc
        from tqdm import tqdm
        def get_auroc_fpr95(true, pred):
            fprs, tprs, thresholds = roc_curve(true, pred)
            roc_auc = auc(fprs, tprs)
            fpr95 = 0
            for i, tpr in enumerate(tqdm(tprs, desc="TPR@FPR=0.95")):
                if tpr >= 0.95:
                    fpr95 = fprs[i]
                    break
            return roc_auc, fpr95

        truths = np.array(self.truths)
        predictions = np.array(self.predictions)
        AuPR = average_precision_score(truths, predictions)
        AuROC, FPR95 = get_auroc_fpr95(truths, predictions)
        locals_ = locals()
        return {k: locals_[k] for k in self.metrics}

    def __repr__(self):
        return f"{type(self).__name__}(metrics={self.metrics})"


class _MeanMetric(AccumulatingMetric, metaclass=ABCMeta):
    def __init__(self, name, value_extractor=None):
        self.name = name
        self.value_extractor = value_extractor or (lambda x: x[name])
        self.reset()

    def reset(self):
        self._sum = KleinSum()
        self._n = EPS

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"


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

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"


class MaxMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, max, extract_func=extract_func)


class MinMetric(_ExtremumMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, min, extract_func=extract_func)


class StatMetric(AccumulatingMetric):
    def __init__(self, name, stat_func, extract_func=None):
        self.name = name
        self.stat_func = stat_func
        self.extract_func = extract_func or (lambda x: x[name])
        self.reset()

    def reset(self):
        self._ext = []

    @torch.no_grad()
    def update(self, iter_result):
        self._ext.append(self.extract_func(iter_result))

    @torch.no_grad()
    def compute(self):
        return {self.name: self.stat_func(self._ext)}

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"


class MedianMetric(StatMetric):
    def __init__(self, name, extract_func=None):
        super().__init__(name, lambda a: np.median(a), extract_func=extract_func)


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

    def __repr__(self):
        return f"{type(self).__name__}({self.metrics})"


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


class MedianMultiMetric(_MultiMetric):
    def __init__(self, filter):
        super().__init__(filter=filter, metric_f=MedianMetric)


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
