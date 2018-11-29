from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from util import disable_sklearn_warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from data.features import *

class RandomVerificator:
    def __init__(self): pass
    def fit(self,positives,negatives):
        pass
    def predict(self,test):
        return np.random.rand()

class AuthorshipVerificator:

    def __init__(self, nfolds=10,
                 params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight':['balanced',None]},
                 estimator=SVC):
        self.nfolds = nfolds
        self.params = params
        if estimator is SVC:
            self.params['kernel'] = ['linear', 'rbf']
            self.probability = True
            self.svm = estimator(probability=self.probability)
        elif estimator is LinearSVC:
            self.probability = False
            self.svm = estimator()
        elif estimator is LogisticRegression:
            self.probability = True
            self.svm = LogisticRegression()

    def fit(self,X,y):
        if not isinstance(y,np.ndarray): y=np.array(y)
        positive_examples = y.sum()
        if positive_examples >= self.nfolds:
            print('optimizing {}'.format(self.svm.__class__.__name__))
            self.estimator = GridSearchCV(self.svm, param_grid=self.params, cv=self.nfolds, scoring=make_scorer(f1_score), n_jobs=-1)
        else:
            self.estimator = self.svm

        self.estimator.fit(X, y)

        if isinstance(self.estimator, GridSearchCV):
            print('Best params: {}'.format(self.estimator.best_params_))
            print('computing the cross-val score')
            f1scores = self.estimator.best_score_
            f1_mean, f1_std = f1scores.mean(), f1scores.std()
            print('F1-measure={:.3f} (+-{:.3f})\n'.format(f1_mean, f1_std))

        return self

    def predict(self, test, epistola_name=''):
        pred = self.estimator.predict(test)
        full_doc_prediction = pred[0]
        print('{} is from the same author: {}'.format(epistola_name, 'Yes' if full_doc_prediction == 1 else 'No'))
        if len(pred) > 1:
            fragment_predictions = pred[1:]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction

    def predict_proba(self, test, epistola_name=''):
        assert self.probability, 'svm is not calibrated'
        pred = self.estimator.predict_proba(test)
        full_doc_prediction = pred[0,1]
        print('{} is from the same author: {}'.format(epistola_name, full_doc_prediction))
        if len(pred) > 1:
            fragment_predictions = pred[1:,1]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction



