import numpy as np


def soft_multiclass_confusion_matrix(true, pred, class_count):
    cm = np.zeros([class_count] * 2)
    for c in range(class_count):
        cm[c, :] = pred[true == c, :].sum(0)
    return cm


def multiclass_confusion_matrix(true, pred, class_count):
    pred = np.eye(C)[np.argmax(pred, 1)]
    cm = np.zeros([class_count] * 2)
    for c in range(class_count):
        cm[c, :] = pred[true == c, :].sum(0)
    return cm


C = 3
N = 20
true = np.array([np.arange(C)] * N).flatten()
true_oh = np.eye(C)[true]
pred = np.abs(np.random.randn(len(true), C)) / 2
pred /= pred.sum(1, keepdims=True)
pred = (4 * pred + true_oh) / 5

np.set_printoptions(precision=1)

cm = multiclass_confusion_matrix(true, pred, C)
print(cm)
cm = soft_multiclass_confusion_matrix(true, pred, C)
print(cm)
cm = soft_multiclass_confusion_matrix(true, true_oh, C)
print(cm)
