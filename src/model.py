from util import disable_sklearn_warnings
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, LeaveOneGroupOut, cross_val_score, GroupKFold, KFold, \
    StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from data.features import *

class RandomVerificator:
    def __init__(self): pass
    def fit(self,positives,negatives):
        pass
    def predict(self,test):
        return np.random.rand()

def get_counters(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "Format not consistent between true and predicted labels."
    nd = len(true_labels)
    tp = np.sum(predicted_labels[true_labels == 1])
    fp = np.sum(predicted_labels[true_labels == 0])
    fn = np.sum(true_labels[predicted_labels == 0])
    tn = nd - (tp+fp+fn)
    return tp,fp,fn,tn

def f1_from_counters(tp,fp,fn,tn):
    num = 2.0 * tp
    den = 2.0 * tp + fp + fn
    if den > 0: return num / den
    # we define f1 to be 1 if den==0 since the classifier has correctly classified all instances as negative
    return 1.0

def f1(true_labels, predicted_labels):
    tp, fp, fn, tn = get_counters(true_labels,predicted_labels)
    return f1_from_counters(tp, fp, fn, tn )


class AuthorshipVerificator:

    def __init__(self, nfolds=10,
                 params = {'C': np.logspace(-4,+4,9), 'class_weight':['balanced',None]},
                 estimator=SVC,
                 author_name=None):
        self.nfolds = nfolds
        self.params = params
        self.author_name = author_name if author_name else 'this author'
        if estimator is SVC:
            self.params['kernel'] = ['linear', 'rbf']
            self.probability = True
            self.classifier = estimator(probability=self.probability)
        elif estimator is LinearSVC:
            self.probability = False
            self.classifier = estimator()
        elif estimator is LogisticRegression:
            self.probability = True
            self.classifier = LogisticRegression()

    def fit(self,X,y,groups=None):
        if not isinstance(y,np.ndarray): y=np.array(y)
        positive_examples = y.sum()
        if positive_examples >= self.nfolds:
            print('optimizing {}'.format(self.classifier.__class__.__name__))
            folds = list(StratifiedKFold(n_splits=self.nfolds).split(X, y))
            self.estimator = GridSearchCV(self.classifier, param_grid=self.params, cv=folds, scoring=make_scorer(f1), n_jobs=-1)
        else:
            self.estimator = self.classifier

        self.estimator.fit(X, y)

        if isinstance(self.estimator, GridSearchCV):
            print('Best params: {}'.format(self.estimator.best_params_))
            print('computing the cross-val score')
            f1scores = self.estimator.best_score_
            f1_mean, f1_std = f1scores.mean(), f1scores.std()
            print('F1-measure={:.3f} (+-{:.3f} cv={})\n'.format(f1_mean, f1_std, f1scores))
            self.estimator = self.estimator.best_estimator_

        return self

    def leave_one_out(self, X, y, files, groups=None, test_lowest_index_only=True, counters=False):

        if groups is None:
            print('Computing LOO without groups')
            folds = list(LeaveOneOut().split(X,y))
        else:
            print('Computing LOO with groups')
            logo = LeaveOneGroupOut()
            folds=list(logo.split(X,y,groups))
            if test_lowest_index_only:
                print('ignoring fragments')
                folds = [(train, np.min(test, keepdims=True)) for train, test in folds]

        scores = cross_val_score(self.estimator, X, y, cv=folds, scoring=make_scorer(f1), n_jobs=-1)
        missclassified = '\n'.join(files[scores==0].tolist())
        print(scores)
        print(missclassified)

        if counters and test_lowest_index_only:
            yfull_true = y[:len(folds)]
            yfull_predict = np.zeros_like(yfull_true)
            yfull_predict[scores == 1] = yfull_true[scores == 1]
            yfull_predict[scores != 1] = 1-yfull_true[scores != 1]
            tp, fp, fn, tn = get_counters(yfull_true, yfull_predict)
            return scores.mean(), scores.std(), tp, fp, fn, tn
        else:
            return scores.mean(), scores.std()

    def predict(self, test, epistola_name=''):
        pred = self.estimator.predict(test)
        full_doc_prediction = pred[0]
        print('{} is from the same author: {}'.format(epistola_name, 'Yes' if full_doc_prediction == 1 else 'No'))
        if len(pred) > 1:
            fragment_predictions = pred[1:]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction, None

    def predict_proba(self, test, epistola_name=''):
        assert self.probability, 'svm is not calibrated'
        pred = self.estimator.predict_proba(test)
        full_doc_prediction = pred[0,1]
        print(f'{epistola_name} is from {self.author_name} with Probability {full_doc_prediction:.3f}')
        if len(pred) > 1:
            fragment_predictions = pred[1:,1]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction, None



