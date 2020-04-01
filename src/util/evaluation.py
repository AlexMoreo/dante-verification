import numpy as np


def get_counters(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "Format not consistent between true and predicted labels."
    nd = len(true_labels)
    tp = np.sum(predicted_labels[true_labels == 1])
    fp = np.sum(predicted_labels[true_labels == 0])
    fn = np.sum(true_labels[predicted_labels == 0])
    tn = nd - (tp+fp+fn)
    return tp, fp, fn, tn


def f1_from_counters(tp, fp, fn, tn):
    num = 2.0 * tp
    den = 2.0 * tp + fp + fn
    if den > 0: return num / den
    # f1 is undefined when den==0; we define f1=1 if den==0 since the classifier has correctly classified all instances as negative
    return 1.0


def f1(true_labels, predicted_labels):
    tp, fp, fn, tn = get_counters(true_labels,predicted_labels)
    return f1_from_counters(tp, fp, fn, tn)