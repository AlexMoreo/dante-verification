from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, LeaveOneGroupOut, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from data.features import *
from util.calibration import CalibratedClassifierCV
from util.evaluation import f1, get_counters


class AuthorshipVerificator:

    def __init__(self, nfolds=10,
                 params={'C': np.logspace(-4, +3, 8)},
                 C=1.,
                 author_name=None):
        self.nfolds = nfolds
        self.params = params
        self.author_name = author_name if author_name else 'this author'
        self.classifier = LogisticRegression(C=C, class_weight='balanced')

    def fit(self, X, y):
        y = np.asarray(y)
        positive_examples = y.sum()
        if positive_examples >= self.nfolds and self.params is not None:
            print('optimizing {}'.format(self.classifier.__class__.__name__))
            folds = list(StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=42).split(X, y))
            self.estimator = GridSearchCV(
                self.classifier, param_grid=self.params, cv=folds, scoring=make_scorer(f1), n_jobs=-1
            )
        else:
            self.estimator = self.classifier

        self.estimator.fit(X, y)

        if isinstance(self.estimator, GridSearchCV):
            f1_mean = self.estimator.best_score_.mean()
            print(f'Best params: {self.estimator.best_params_} (cross-validation F1={f1_mean:.3f})')
            self.estimator = self.estimator.best_estimator_

        self.estimator = CalibratedClassifierCV(base_estimator=self.estimator, cv=self.nfolds, ensemble=False)
        self.estimator.fit(X, y)

        return self

    def leave_one_out(self, X, y, files, groups=None, test_lowest_index_only=True, counters=False):
        if groups is None:
            print('Computing LOO without groups')
            folds = list(LeaveOneOut().split(X, y))
        else:
            print('Computing LOO with groups')
            logo = LeaveOneGroupOut()
            folds = list(logo.split(X, y, groups))
            if test_lowest_index_only:
                print('ignoring fragments')
                folds = [(train, np.min(test, keepdims=True)) for train, test in folds]

        scores = cross_val_score(self.estimator, X, y, cv=folds, scoring=make_scorer(f1), n_jobs=-1)
        missclassified = files[scores == 0].tolist()
        #if hasattr(self.estimator, 'predict_proba') and len(missclassified) > 0:
        #    missclassified_prob = self.estimator.predict_proba(csr_matrix(X)[scores == 0])[:, 1]
        #    missclassified_prob = missclassified_prob.flatten().tolist()
        #    missclassified = [f'{file} Pr={prob:.3f}' for file, prob in zip(missclassified,missclassified_prob)]
        print('missclassified texts:')
        print('\n'.join(missclassified))


        if counters and test_lowest_index_only:
            yfull_true = y[:len(folds)]
            yfull_predict = np.zeros_like(yfull_true)
            yfull_predict[scores == 1] = yfull_true[scores == 1]
            yfull_predict[scores != 1] = 1-yfull_true[scores != 1]
            tp, fp, fn, tn = get_counters(yfull_true, yfull_predict)
            return scores.mean(), scores.std(), tp, fp, fn, tn
        else:
            return scores.mean(), scores.std()

    def predict(self, test):
        pred = self.estimator.predict(test)
        full_doc_prediction = pred[0]
        if len(pred) > 1:
            fragment_predictions = pred[1:]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction

    def predict_proba(self, test):
        assert hasattr(self, 'predict_proba'), 'the classifier is not calibrated'
        pred = self.estimator.predict_proba(test)
        full_doc_prediction = pred[0,1]
        if len(pred) > 1:
            fragment_predictions = pred[1:,1]
            print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
            return full_doc_prediction, fragment_predictions
        return full_doc_prediction, []



