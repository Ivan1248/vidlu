import numpy as np
from sklearn.metrics import confusion_matrix

EPS = 1e-8


# from dlu #########################################################################################


class AccumulatingMetric:
    def reset(self):
        raise NotImplementedError()

    def update(self, iter_output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


class FuncMetric(AccumulatingMetric):
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or type(self).__name__
        self.reset()

    def reset(self):
        self._sum = 0
        self._n = EPS

    def update(self, iter_output):
        self._sum += self.func(iter_output)
        self._n += 1

    def compute(self):
        return {self.name: self._sum / self._n}


class ClassificationMetrics(AccumulatingMetric):
    def __init__(self, class_count):
        self.class_count = class_count
        self.cm = np.zeros([class_count] * 2)
        self.labels = np.arange(class_count)
        self.active = False

    def reset(self):
        self.cm.fill(0)

    def update(self, iter_output):
        pred = iter_output.target.flatten().int().cpu().numpy()
        true = iter_output.outputs.hard_prediction.flatten().int().cpu().numpy()
        self.cm += confusion_matrix(pred, true, labels=self.labels)

    def compute(self, returns=('A', 'mP', 'mR', 'mF1', 'mIoU'), eps=1e-8):
        # Computes macro-averaged classification evaluation metrics based on the
        # accumulated confusion matrix and clears the confusion matrix.
        tp = np.diag(self.cm)
        actual_pos = self.cm.sum(axis=1)
        pos = self.cm.sum(axis=0) + eps
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
