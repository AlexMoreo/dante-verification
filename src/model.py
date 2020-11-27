from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from data.features import *
from util.evaluation import f1_metric
from typing import List, Union



class AuthorshipVerificator(BaseEstimator):

    def __init__(self, nfolds=10, param_grid=None, learner=None, C=1., alpha=0.001, class_weight='balanced',
                 random_seed=41, feat_selection_slices=None, feat_selection_ratio=1):
        self.nfolds = nfolds
        self.param_grid = param_grid
        self.learner = learner
        self.C = C
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_seed = random_seed
        self.feat_selection_slices = feat_selection_slices
        self.feat_selection_ratio = feat_selection_ratio

    def fit(self, X, y, groups=None, hyperparam_optimization=True):
        if self.param_grid is None and hyperparam_optimization:
            raise ValueError('Param grid is None, but hyperparameter optimization is requested')

        if self.feat_selection_slices is not None:
            self.fs = MultiRangeFeatureSelector(self.feat_selection_slices, feat_sel=self.feat_selection_ratio)
            X = self.fs.fit(X, y).transform(X)

        if self.learner == 'lr':
            self.classifier = LogisticRegression(
                C=self.C, class_weight=self.class_weight, max_iter=1000, random_state=self.random_seed, solver='lbfgs'
            )
        elif self.learner == 'svm':
            self.classifier = LinearSVC(C=self.C, class_weight=self.class_weight)
        elif self.learner == 'mnb':
            self.classifier = MultinomialNB(alpha=self.alpha)

        y = np.asarray(y)
        positive_examples = y.sum()

        if groups is None:
            groups = np.arange(len(y))

        if hyperparam_optimization and (positive_examples >= self.nfolds) and (len(np.unique(groups[y==1])) > 1):
            folds = list(GroupKFold(n_splits=self.nfolds).split(X, y, groups))
            self.estimator = GridSearchCV(
                self.classifier, param_grid=self.param_grid, cv=folds, scoring=make_scorer(f1_metric), n_jobs=-1,
                refit=True, error_score=0
            )
        else:
            # insufficient positive examples or document groups for grid-search; using default classifier
            print('insufficient positive examples or document groups for grid-search; using default classifier')
            self.estimator = self.classifier

        self.estimator.fit(X, y)

        if isinstance(self.estimator, GridSearchCV):
            f1_mean = self.estimator.best_score_.mean()
            self.choosen_params_ = self.estimator.best_params_
            print(f'Best params: {self.choosen_params_} (cross-validation F1={f1_mean:.3f})')
        else:
            self.choosen_params_ = {'C': self.C, 'alpha': self.alpha}

        return self

    def predict(self, test):
        if self.feat_selection_slices is not None:
            test = self.fs.transform(test)
        return self.estimator.predict(test)

    def predict_proba(self, test):
        assert hasattr(self, 'predict_proba'), 'the classifier is not calibrated'
        if self.feat_selection_slices is not None:
            test = self.fs.transform(test)
        prob = self.estimator.predict_proba(test)
        return prob


class RangeFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, range: slice, feat_sel: Union[float, int]):
        self.range = range
        self.feat_sel = feat_sel

    def fit(self, X, y):
        nF = self.range.stop-self.range.start
        if isinstance(self.feat_sel, int) and self.feat_sel>0:
            num_feats = self.feat_sel
        elif isinstance(self.feat_sel, float) and 0. <= self.feat_sel <= 1.:
            num_feats = int(self.feat_sel * nF)
        else:
            raise ValueError('feat_sel should be a positive integer or a float in [0,1]')
        self.selector = SelectKBest(chi2, k=num_feats)
        self.selector.fit(X[:,self.range], y)
        return self

    def transform(self, X):
        Z = self.selector.transform(X[:,self.range])
        normalize(Z, norm='l2', copy=False)
        X = csr_matrix(hstack([X[:,:self.range.start], Z, X[:,self.range.stop:]]))
        return X


class MultiRangeFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, ranges: List[slice], feat_sel: Union[float,int]):
        self.ranges = ranges
        self.feat_sel = feat_sel

    def fit(self, X, y):
        assert isinstance(self.ranges, list), 'ranges should be a list of slices'
        self.__check_ranges_collisions(self.ranges)
        self.ranges = self.__sort_ranges(self.ranges)
        self.selectors = [RangeFeatureSelector(r, self.feat_sel).fit(X, y) for r in self.ranges]
        return self

    def transform(self, X):
        for selector in self.selectors:
            X = selector.transform(X)
        return X

    def __check_ranges_collisions(self, ranges: List[slice]):
        for i,range_i in enumerate(ranges):
            for j,range_j in enumerate(ranges):
                if i==j: continue
                if range_i.start <= range_j.start <= range_i.stop: return False
                if range_i.start <= range_j.stop <= range_i.stop: return False
        return True

    def __sort_ranges(self, ranges: List[slice]):
        return np.asarray(ranges)[np.argsort([r.start for r in ranges])[::-1]]


def get_valid_folds(nfolds, X, y, groups, max_trials=100):
    trials = 0
    folds = list(GroupKFold(n_splits=nfolds).split(X, y, groups))
    n_docs = len(y)
    print(f'different classes={np.unique(y)}; #different documents={len(np.unique(groups))} positives={len(np.unique(groups[y==1]))}')
    while any(len(np.unique(y[train])) < 2 for train, test in folds):
        shuffle_index = np.random.permutation(n_docs)
        X, y, groups = X[shuffle_index], y[shuffle_index], groups[shuffle_index]
        folds = list(GroupKFold(n_splits=nfolds).split(X, y, groups))
        print(f'\ttrial{trials}:{[len(np.unique(y[train])) for train, test in folds]}')
        trials+=1
        if trials>max_trials:
            raise ValueError(f'could not meet condition after {max_trials} trials')
    return folds
