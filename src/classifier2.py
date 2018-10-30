from sklearn.svm import *
from sklearn.model_selection import cross_val_score, GridSearchCV
from doc_representation2 import *
from sklearn.metrics import f1_score, make_scorer

probability=False
# SVM = SVC
SVM = LinearSVC

nfolds = 3
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight':['balanced',None]}
if SVM is SVC:
    params['kernel']=['linear','rbf']

path = '../testi'

Xtr,ytr,ep1,ep2 = load_documents(path, split_documents=True, function_words_freq=True, tfidf=True, tfidf_feat_selection_ratio=0.1)

# learn a SVM

# svm = SVM(probability=probability)
svm = SVM()

positive_examples = ytr.sum()
if positive_examples>nfolds:
    print('optimizing {}'.format(svm.__class__.__name__))
    svm = GridSearchCV(svm, param_grid=params, cv=nfolds, scoring=make_scorer(f1_score))

svm.fit(Xtr, ytr)

if isinstance(svm, GridSearchCV):
    print('Best params: {}'.format(svm.best_params_))

# evaluation of results
print('computing the cross-val score')
f1scores = cross_val_score(svm, Xtr, ytr, cv=nfolds, n_jobs=-1, scoring=make_scorer(f1_score))
f1_mean, f1_std = f1scores.mean(), f1scores.std()
print('F1-measure={:.3f} (+-{:.3f})\n'.format(f1_mean, f1_std))

# final test
def predictEpistola(ep, epistola_name):
    pred = svm.predict(ep)
    full_doc_prediction = pred[0]
    print('{} is from Dante: {}'.format(epistola_name, 'Yes' if full_doc_prediction == 1 else 'No'))
    if len(pred>0):
        fragment_predictions= pred[1:]
        print('fragments average {:.3f}, array={}'.format(fragment_predictions.mean(), fragment_predictions))
        if SVM is SVC and probability:
            prob = svm.predict_proba(ep)[:,1]
            np.set_printoptions(precision=2, linewidth=200)
            print('probabilistic view: full={:.3f}, fragments average {:.3f}, fragments={}'.format(prob[0], prob[1:].mean(), prob[1:]))

print('Predicting the Epistolas')
predictEpistola(ep1, 'Epistola 1')
predictEpistola(ep2, 'Epistola 2')
