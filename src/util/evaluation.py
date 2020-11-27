import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from joblib import Parallel, delayed


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


def f1_metric(true_labels, predicted_labels):
    tp, fp, fn, tn = get_counters(true_labels,predicted_labels)
    return f1_from_counters(tp, fp, fn, tn)


def leave_one_out(model, X, y, files, groups):
    print(f'Computing LOO with groups over {X.shape[0]} documents')
    logo = LeaveOneGroupOut()

    # Fragments are ignored in the test; only full documents are evaluated.
    # The index of the full document is the lowest index
    folds = [(train, np.min(test, keepdims=True)) for train, test in logo.split(X, y, groups)]

    def _classify_held_out(train, test, X, y, model):
        X = csr_matrix(X)
        # hyperparam_optim = (len(np.unique(groups[y[train] == 1])) > 2)
        model.fit(X[train], y[train], groups[train])#, hyperparam_optim=hyperparam_optim)
        y_pred = model.predict(X[test]).item()
        score = (y_pred == y[test]).item()
        return y_pred, score

    predictions_scores = Parallel(n_jobs=-1)(
        delayed(_classify_held_out)(train, test, X, y, model) for train, test in folds
        #tqdm(
        #    folds, desc=f'optimizing via grid search each of the {len(folds)} prediction problems'
        #)
    )
    predictions = np.asarray([p for p,s in predictions_scores])
    scores = np.asarray([s for p, s in predictions_scores])
    print(predictions, scores)

    missclassified = files[scores == 0].tolist()

    yfull_true = y[:len(folds)]
    tp, fp, fn, tn = get_counters(yfull_true, predictions)
    f1 = f1_from_counters(tp, fp, fn, tn)
    acc = scores.mean()

    return acc, f1, tp, fp, fn, tn, missclassified