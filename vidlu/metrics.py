import typing as T
from abc import ABCMeta
from functools import wraps

import numpy as np
import torch

from vidlu.ops import one_hot
from vidlu.torch_utils import retry_if_cuda_oom
from vidlu.utils.collections import NameDict
from vidlu.utils.num import KleinSum

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
    if loop_version is None and true.device.type == 'cuda':
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


# Metric name grammar ##############################################################################

#: Suffix of a support-restricted metric name: 'mF1_supp10' is the mean of the per-class F1 over
#: the classes with at least 10 ground-truth examples.
SUPPORT_SUFFIX = "_supp"

#: Mean-over-classes metric name -> the per-class array it is the mean of. Only these can carry
#: a support suffix.
MACRO_METRIC_TO_PER_CLASS = {"mP": "P", "mR": "R", "mF1": "F1", "mIoU": "IoU",
                             "mNLL": "cNLL", "mBrier": "cBrier"}

#: Metrics that exist only with a support suffix: 'nc_suppN' is the number of classes with
#: support >= N, without which a restricted mean cannot be read.
SUPPORT_ONLY_METRICS = frozenset({"nc"})


def parse_support_suffix(name: str) -> tuple[str, int | None]:
    """Splits 'mF1_supp10' into ('mF1', 10); a name without the suffix gives (name, None)."""
    base, separator, threshold = name.partition(SUPPORT_SUFFIX)
    if not separator:
        return name, None
    if not threshold.isdigit():
        raise ValueError(f"Metric {name!r} has a non-integer support threshold {threshold!r}.")
    return base, int(threshold)


def macro_over_supported(per_class_values, support, min_support: int) -> torch.Tensor:
    """Mean of per-class values over the classes with `support >= min_support`.

    Args:
        per_class_values: (C,) or (B, C) per-class metric values.
        support: Per-class ground-truth example counts of the same shape.
        min_support: Minimum number of ground-truth examples for a class to enter the mean.

    Returns:
        The mean over qualifying classes (a scalar, or (B,) for a batch); NaN where no class
        qualifies.
    """
    per_class_values = torch.as_tensor(per_class_values)
    if not per_class_values.is_floating_point():
        per_class_values = per_class_values.double()
    mask = torch.as_tensor(support) >= min_support
    return torch.where(mask, per_class_values, 0).sum(-1) / mask.sum(-1)


def select_metrics(metrics: T.Mapping[str, torch.Tensor], support: torch.Tensor, names):
    """Picks `names` from computed `metrics`, deriving the support-restricted ones.

    A name not in `metrics` must be `<macro>_suppN` for a macro metric whose per-class array
    is in `metrics`, or `nc_suppN`. Anything else raises rather than being skipped, so a
    misspelled metric cannot look like one that never fired.

    Args:
        metrics: Base metric name -> value, including the per-class arrays.
        support: Per-class ground-truth counts, (C,) or (B, C) like the per-class arrays.
        names: The metric names to return.
    """
    result = {}
    for name in names:
        if name in metrics:
            result[name] = metrics[name]
            continue
        base, min_support = parse_support_suffix(name)
        if min_support is None:
            if base in SUPPORT_ONLY_METRICS:
                raise ValueError(f"Metric {name!r} exists only with a support threshold,"
                                 f" e.g. {base}{SUPPORT_SUFFIX}10.")
            raise ValueError(f"Unknown metric {name!r}. Available: {sorted(metrics)}.")
        if base in SUPPORT_ONLY_METRICS:
            result[name] = (support >= min_support).sum(-1)
        elif base in MACRO_METRIC_TO_PER_CLASS and MACRO_METRIC_TO_PER_CLASS[base] in metrics:
            result[name] = macro_over_supported(metrics[MACRO_METRIC_TO_PER_CLASS[base]],
                                                support, min_support)
        else:
            restrictable = sorted(m for m, c in MACRO_METRIC_TO_PER_CLASS.items() if c in metrics)
            raise ValueError(
                f"Metric {name!r} restricts {base!r} by support, but {base!r} is not a mean over"
                f" classes available here. Thresholds apply to {restrictable} and"
                f" {sorted(SUPPORT_ONLY_METRICS)}.")
    return result


def confusion_matrix_class_stats(cm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-class counts from a (C, C) confusion matrix with rows = ground truth.

    Returns:
        'tp' (diagonal), 'pos' (predicted positives, column sums), 'actual' (ground-truth
        positives, row sums).
    """
    return dict(tp=cm.diagonal(), pos=cm.sum(0), actual=cm.sum(1))


def classification_metrics(cm, returns=('A', 'mP', 'mR', 'mF1', 'mIoU', 'cm'), eps=1e-8, ignore_missing_classes=False):
    """Computes macro-averaged classification evaluation metrics based on the
    accumulated confusion matrix.

    Supports batches (when `cm` is a batch of matrices).

    **Important**: By default, macro-averaged metrics (mP, mR, mF1, mIoU) are 
    computed over all classes, including classes with no samples (metric=0).
    
    This differs from sklearn's `f1_score(average='macro')`, which only averages
    over classes that appear in `y_true` or `y_pred`. Set `ignore_missing_classes=True` 
    to match sklearn's behavior.
    
    Args:
        cm (Tensor): A confusion matrix of shape (C, C) or batch (B, C, C).
        returns (Sequence): A list of metrics that should be returned.
        eps (float): A number to add to the denominator to avoid division by 0.
        ignore_missing_classes (bool): If True, compute macro metrics only over classes 
            with samples (matches sklearn). Default False.

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
    mask = (actual_pos + fp) > 0 if ignore_missing_classes else None  
    metrics.update({'m' + k: masked_mean(v, mask) for k, v in metrics.items()})
    metrics['cm'] = cm
    metrics['A'] = tp.sum(dim=is_batch) / pos.sum(dim=is_batch)
    metrics['num_correct'] = tp.sum(dim=is_batch)
    metrics['n'] = pos.sum(dim=is_batch)  # number of (non-ignore) examples

    # Chance-corrected agreement. MCC is NaN (0/0) when every ground-truth or every predicted
    # label is one class, kappa when both are; sklearn returns 0 there, which is
    # indistinguishable from chance-level performance, so NaN is kept.
    n = pos.sum(dim=is_batch).double()
    num_correct = tp.sum(dim=is_batch).double()
    pos_d, actual_d = pos.double(), actual_pos.double()
    s = (pos_d * actual_d).sum(dim=is_batch)  # sum_k pred_k * actual_k
    numerator = num_correct * n - s
    metrics['kappa'] = (numerator / (n ** 2 - s)).float()
    metrics['MCC'] = (numerator / torch.sqrt((n ** 2 - (pos_d ** 2).sum(dim=is_batch))
                                             * (n ** 2 - (actual_d ** 2).sum(dim=is_batch)))
                      ).float()
    if isinstance(returns, str):
        return select_metrics(metrics, actual_pos, (returns,))[returns]
    return select_metrics(metrics, actual_pos, returns)


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
    - 'n': number of (non-ignore) examples accumulated in the confusion matrix
    - 'kappa': Cohen's kappa (chance-corrected accuracy); NaN where undefined
    - 'MCC': multi-class Matthews correlation coefficient (Gorodkin's R_k); NaN where undefined
    - 'cm': confusion matrix
    - '<macro>_suppN' (e.g. 'mF1_supp10'): the mean over the classes with at least N
      ground-truth examples; NaN if none qualifies. 'nc_suppN': how many classes qualify.

    **Important**: By default, macro-averaged metrics (mP, mR, mF1, mIoU) are
    computed over all classes. This differs from sklearn's `average='macro'`, which only averages
    over classes that appear in `y_true` or `y_pred`. Set `ignore_missing_classes=True` 
    to match sklearn's behavior.

    Args:
        class_count: Number of classes.
        get_target, get_hard_prediction: Extract the (N,) integer targets and the (N,) hard
            predictions from the iteration result.
        metrics: Names of the metrics to return from `compute`.
    """
    def __init__(self, class_count, get_target=lambda r: r.target,
                 get_hard_prediction=lambda r: r.out.argmax(1),
                 metrics=('A', 'mP', 'mR', 'mIoU'), device=None, cm=None,
                 ignore_missing_classes=False):
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
        self.ignore_missing_classes = ignore_missing_classes

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
    def compute(self, eps=1e-8, metrics=None):
        """Computes `metrics` (default: the configured ones) from the confusion matrix."""
        computed = classification_metrics(
            self.cm, returns=self.metrics if metrics is None else metrics, eps=eps,
            ignore_missing_classes=self.ignore_missing_classes)
        return {k: v.item() if v.dim() == 0 else v.cpu().numpy().copy()
                for k, v in computed.items()}

    def __repr__(self):
        return f"{type(self).__name__}(class_count={self.class_count}, metrics={self.metrics})"


def _nanmean_over_classes(per_class: torch.Tensor) -> torch.Tensor:
    """Mean over the classes whose value is defined; NaN if none is."""
    defined = ~torch.isnan(per_class)
    return per_class[defined].mean() if defined.any() else per_class.new_tensor(float("nan"))


class ProbabilisticClassificationMetrics(AccumulatingMetric):
    """Computes proper scoring rules from predicted class distributions.

    Unlike `ClassificationMetrics`, which sees only the hard prediction, this accumulates
    per-class sums of two per-sample scores over all non-ignored (`target != -1`) examples:

    - negative log-likelihood, `-log p[target]`;
    - the multi-class Brier score, `sum_k (p_k - 1[k == target])^2` (Brier, 1950), in [0, 2].

    Supported metrics:
    - 'NLL', 'Brier': means over examples
    - 'cNLL', 'cBrier': per-class means (NaN for a class with no ground-truth example)
    - 'mNLL', 'mBrier': means of the per-class means over the classes with ground truth, i.e.
      class-balanced versions that weight every class equally
    - 'mNLL_suppN', 'mBrier_suppN', 'nc_suppN': the same restricted to classes with at least N
      ground-truth examples, and the number of such classes
    - 'n': number of (non-ignore) examples

    The bare names are means over examples, unlike 'P'/'R'/'F1', whose bare names are
    per-class arrays because their micro variant is just accuracy; the per-class arrays
    therefore carry the 'c' prefix, and 'm' keeps its meaning of a mean over classes.

    Args:
        class_count: Number of classes.
        output_kind: What `get_output` returns: 'logits' (log-softmax is applied) or 'probs'
            (probabilities, whose log is taken).
        get_target, get_output: Extract the (N,) integer targets and the (N, C) outputs from
            the iteration result.
        metrics: Names of the metrics to return from `compute`.
    """

    OUTPUT_KINDS = ('logits', 'probs')

    def __init__(self, class_count, output_kind: T.Literal['logits', 'probs'],
                 get_target=lambda r: r.target, get_output=lambda r: r.out,
                 metrics=('NLL', 'Brier', 'mNLL', 'mBrier'), device=None):
        if output_kind not in self.OUTPUT_KINDS:
            raise ValueError(
                f"output_kind must be one of {self.OUTPUT_KINDS}, got {output_kind!r}.")
        self.class_count = class_count
        self.output_kind = output_kind
        self.get_target = get_target
        self.get_output = get_output
        self.metrics = metrics
        self.nll_sum = torch.zeros(class_count, dtype=torch.float64, device=device)
        self.brier_sum = torch.zeros(class_count, dtype=torch.float64, device=device)
        self.count = torch.zeros(class_count, dtype=torch.int64, device=device)

    @torch.no_grad()
    def reset(self):
        for x in (self.nll_sum, self.brier_sum, self.count):
            x.fill_(0)

    def _log_probs(self, out: torch.Tensor) -> torch.Tensor:
        out = out.double()
        return out.log_softmax(1) if self.output_kind == 'logits' else out.log()

    @torch.no_grad()
    def update(self, iter_result):
        true = self.get_target(iter_result).flatten()
        out = self.get_output(iter_result)
        if out.shape != (true.shape[0], self.class_count):
            raise ValueError(f"Expected outputs of shape {(true.shape[0], self.class_count)},"
                             f" got {tuple(out.shape)}.")
        non_ignored = true != -1
        true, out = true[non_ignored], out[non_ignored]
        if self.count.device != out.device:
            self.nll_sum, self.brier_sum, self.count = (
                x.to(out.device) for x in (self.nll_sum, self.brier_sum, self.count))
        log_p = self._log_probs(out)
        nll = -log_p.gather(1, true[:, None]).squeeze(1)
        brier = (log_p.exp() - one_hot(true, self.class_count, dtype=log_p.dtype)).pow(2).sum(1)
        self.nll_sum.index_add_(0, true, nll)
        self.brier_sum.index_add_(0, true, brier)
        self.count += torch.bincount(true, minlength=self.class_count)

    @torch.no_grad()
    def compute(self, metrics=None):
        """Computes `metrics` (default: the configured ones) from the accumulated sums."""
        n = self.count.sum()
        count = self.count.double()
        cNLL, cBrier = self.nll_sum / count, self.brier_sum / count  # NaN where count == 0
        all_metrics = dict(
            NLL=self.nll_sum.sum() / n, Brier=self.brier_sum.sum() / n,
            cNLL=cNLL, cBrier=cBrier,
            mNLL=_nanmean_over_classes(cNLL), mBrier=_nanmean_over_classes(cBrier),
            n=n)
        selected = select_metrics(all_metrics, self.count,
                                  self.metrics if metrics is None else metrics)
        return {k: (v.item() if v.dim() == 0 else v.cpu().numpy().copy())
                for k, v in selected.items()}

    def __repr__(self):
        return (f"{type(self).__name__}(class_count={self.class_count},"
                f" output_kind={self.output_kind!r}, metrics={self.metrics})")


# Multi-attribute classification ###################################################################

#: What the outputs handed to a probabilistic metric are: 'logits' (log-softmax is applied),
#: 'probs' (probabilities; the log is taken) or 'hard' (one-hot pseudo-logits, e.g. parsed from
#: generated text, for which probabilistic metrics are meaningless and therefore refused).
OutputKind = T.Literal["logits", "probs", "hard"]


class AttributeSpec(T.NamedTuple):
    index: int
    class_count: int


class ParsedMetricName(T.NamedTuple):
    is_averaged: bool  # has the a prefix
    name: str  # full name
    base: str  # name without the 'a' prefix and suffix, e.g. 'mF1'
    min_support: int | None  # support threshold if the name has a '_suppN' suffix


def parse_metric_name(name: str, known_base_metrics) -> ParsedMetricName:
    """Parses an `[a]<base>[_suppN]` metric name.

    'a' means averaged over attributes; '_suppN' restricts a mean over classes to the classes
    with at least N ground-truth examples.

    Raises:
        ValueError: For an unknown base name, a threshold on a metric that is not a mean over
            classes, a non-integer threshold, or a support-only metric without a threshold.
    """
    base, min_support = parse_support_suffix(name)
    averaged, clean = False, name
    if base not in known_base_metrics and base.startswith("a") \
            and base[1:] in known_base_metrics:
        averaged, base, clean = True, base[1:], clean[1:]
    if base not in known_base_metrics:
        raise ValueError(
            f"Unknown metric {name!r}. Known base names: {sorted(known_base_metrics)}.")
    if min_support is None and base in SUPPORT_ONLY_METRICS:
        raise ValueError(f"Metric {name!r} exists only with a support threshold,"
                         f" e.g. {base}{SUPPORT_SUFFIX}10.")
    if min_support is not None and base not in MACRO_METRIC_TO_PER_CLASS \
            and base not in SUPPORT_ONLY_METRICS:
        raise ValueError(
            f"Metric {name!r} restricts {base!r} by support, but {base!r} is not a mean over"
            f" classes. Thresholds apply to {sorted(MACRO_METRIC_TO_PER_CLASS)} and"
            f" {sorted(SUPPORT_ONLY_METRICS)}.")
    return ParsedMetricName(is_averaged=averaged, name=clean, base=base, min_support=min_support)


class MultiAttributeClassificationMetrics(AccumulatingMetric):
    """Classification metrics for several categorical attributes of one input.

    Each attribute (a categorical property of the entity the input depicts, e.g. the number
    of lanes and the speed limit of a road segment) has its own class set, its own (B, C_i)
    outputs and its own column in the (B, A) integer targets, where -1 marks a missing
    label. Per attribute, a `ClassificationMetrics` accumulates the confusion matrix and,
    when a probabilistic metric is requested, a `ProbabilisticClassificationMetrics`
    accumulates the proper scoring rules.

    Metric names follow `parse_metric_name`:
        - `{X}`: a dict mapping each attribute key to its value of `X`.
        - `a{X}`: the mean of `X` over attributes (a scalar).
        - `{X}_suppN`, `nc_suppN`: the per-class mean `X` restricted to classes with at least
          N ground-truth examples, and the number of such classes.

    An attribute whose value is NaN (undefined, e.g. `MCC` when every label is one class, or
    a restricted mean with no qualifying class) is dropped from `a{X}`; the average is NaN
    only if no attribute has a defined value.

    Args:
        attributes: Ordered mapping from an attribute key (a name or an index; it is the key
            of the per-attribute result dicts) to its `AttributeSpec` (or `(index,
            class_count)` pair).
        metrics: The metric names to compute, or a mapping from the key each value gets in
            the result dict to the metric name (for a key the reporting layer treats
            specially, e.g. `vidlu.experiments.console_hidden("mF1")`); a sequence means
            each name is its own key.
        output_kind: What `get_outputs` returns; see `OutputKind`. A probabilistic metric with
            'hard' outputs raises.
        get_target: Extracts the (B, A) targets from the iteration result.
        get_outputs: Extracts the sequence of (B, C_i) outputs from the iteration result, or
            None when the step produced no outputs (the update is then skipped). For a
            (B, A, C) tensor use `lambda r: r.out.unbind(1)`.
        ignore_missing_classes: Passed to `ClassificationMetrics`.
    """

    #: Base names `ClassificationMetrics` provides.
    CONFUSION_MATRIX_METRICS = frozenset(
        {"A", "mP", "mR", "mF1", "mIoU", "P", "R", "F1", "IoU", "n", "MCC", "kappa"})
    #: Base names that need predicted class distributions (`ProbabilisticClassificationMetrics`).
    PROBABILISTIC_METRICS = frozenset({"NLL", "Brier", "cNLL", "cBrier", "mNLL", "mBrier"})
    KNOWN_BASE_METRICS = CONFUSION_MATRIX_METRICS | PROBABILISTIC_METRICS | SUPPORT_ONLY_METRICS

    def __init__(self, attributes: T.Mapping[T.Hashable, AttributeSpec | tuple[int, int]],
                 metrics: T.Sequence[str] | T.Mapping[str, str],
                 output_kind: OutputKind = "logits",
                 get_target=lambda r: r.target, get_outputs=lambda r: r.out,
                 ignore_missing_classes: bool = False):
        self.attributes = {k: s if isinstance(s, AttributeSpec) else AttributeSpec(*s)
                           for k, s in dict(attributes).items()}
        if not self.attributes:
            raise ValueError("At least one attribute is required.")
        self.metrics = metrics
        self.output_kind = output_kind
        self.get_target = get_target
        self.get_outputs = get_outputs
        self._parsed = self._parse(metrics)
        cm_names, prob_names = self._child_metric_names(self._parsed.values())
        if prob_names and output_kind == "hard":
            raise ValueError(
                f"Metrics {list(prob_names)} need predicted class distributions, but"
                f" output_kind='hard' says the outputs are one-hot. Either drop them or"
                f" evaluate a model that outputs logits or probabilities.")
        # The children read `target` and `out` of the per-attribute slice built in `update`.
        self.attr_to_cm_metrics = {
            a: ClassificationMetrics(class_count=spec.class_count, metrics=cm_names,
                                     ignore_missing_classes=ignore_missing_classes)
            for a, spec in self.attributes.items()}
        # Probabilistic children exist only when a requested metric needs them.
        self.attr_to_prob_metrics = {
            a: ProbabilisticClassificationMetrics(class_count=spec.class_count,
                                                  output_kind=output_kind, metrics=prob_names)
            for a, spec in self.attributes.items()} if prob_names else {}

    @classmethod
    def _parse(cls, metrics: T.Sequence[str] | T.Mapping[str, str]
               ) -> dict[str, ParsedMetricName]:
        """Result key -> parsed name; a sequence of names uses each name as its own key."""
        key_to_name = metrics.items() if isinstance(metrics, T.Mapping) else zip(metrics, metrics)
        return {key: parse_metric_name(name, cls.KNOWN_BASE_METRICS) for key, name in key_to_name}

    @classmethod
    def _child_metric_names(cls, parsed: T.Iterable[ParsedMetricName]) -> tuple[tuple, tuple]:
        """The per-attribute names each kind of child has to compute; `nc` is confusion-matrix."""
        cm_names, prob_names = set(), set()
        for p in parsed:
            (prob_names if p.base in cls.PROBABILISTIC_METRICS else cm_names).add(p.name)
        return tuple(sorted(cm_names)), tuple(sorted(prob_names))

    def _children(self, attr):
        yield self.attr_to_cm_metrics[attr]
        if attr in self.attr_to_prob_metrics:
            yield self.attr_to_prob_metrics[attr]

    def reset(self):
        for a in self.attributes:
            for m in self._children(a):
                m.reset()

    @torch.no_grad()
    def update(self, iter_result):
        outs = self.get_outputs(iter_result)
        if outs is None:
            return
        target = self.get_target(iter_result)
        for a, spec in self.attributes.items():
            if spec.index >= len(outs):
                raise ValueError(f"Attribute {a!r} has output index {spec.index}, but only"
                                 f" {len(outs)} outputs were given.")
            sliced = NameDict(target=target[:, spec.index], out=outs[spec.index])
            for m in self._children(a):
                m.update(sliced)

    def get_confusion_matrices(self) -> dict[T.Hashable, torch.Tensor]:
        """The accumulated (C_i, C_i) confusion matrix of every attribute, rows = ground truth."""
        return {a: m.cm for a, m in self.attr_to_cm_metrics.items()}

    @torch.no_grad()
    def compute(self, metrics=None):
        """Computes `metrics` (default: the configured ones; a sequence or a key-to-name
        mapping as in the constructor).

        Returns:
            A dict with one entry per requested metric, under its key: a scalar for an
            averaged metric (`a{X}`), else a dict mapping each attribute key to its value.
        """
        parsed = self._parsed if metrics is None else self._parse(metrics)
        cm_names, prob_names = self._child_metric_names(parsed.values())
        if prob_names and not self.attr_to_prob_metrics:
            raise ValueError(f"Metrics {list(prob_names)} need predicted class distributions,"
                             f" but none was requested at construction, so none was accumulated.")
        attr_results = {}
        for a in self.attributes:
            attr_results[a] = self.attr_to_cm_metrics[a].compute(metrics=cm_names)
            if prob_names:
                attr_results[a].update(self.attr_to_prob_metrics[a].compute(metrics=prob_names))

        results = {}
        for key, p in parsed.items():
            per_attr = {a: r[p.name] for a, r in attr_results.items()}
            if not p.is_averaged:
                results[key] = per_attr
                continue
            values = list(per_attr.values())
            if any(isinstance(v, np.ndarray) and v.ndim > 0 for v in values):
                raise ValueError(f"Cannot average per-class metric {p.name!r} over attributes.")
            # An undefined (NaN) attribute value is dropped rather than counted as zero; no
            # defined value at all is reported as NaN, not as a score of 0.
            defined = [float(v) for v in values if not np.isnan(v)]
            results[key] = float(np.mean(defined)) if defined else float("nan")
        return results

    def __repr__(self):
        return (f"{type(self).__name__}(attributes={list(self.attributes)},"
                f" metrics={self.metrics}, output_kind={self.output_kind!r})")


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
        from sklearn.metrics import auc, average_precision_score, roc_curve
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
