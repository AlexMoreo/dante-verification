from sklearn.svm import *
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from doc_representation2 import *
import numpy as np

nfolds = 2
do_feat_selection = False
params = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['balanced',None]}]

path = 'Data'
Xtr,ytr,ep2 = load_documents(path)

if do_feat_selection:
    print('feature selection')
    num_feats = int(0.1 * Xtr.shape[1])
    feature_selector = SelectKBest(chi2, k=num_feats)
    Xtr = feature_selector.fit_transform(Xtr,ytr)
    print('final shape={}'.format(Xtr.shape))
        #ep1 = feature_selector.transform(ep1)
    ep2 = feature_selector.transform(ep2)


# learn a SVM
print('optimizing a SVM')
svm_base = LinearSVC(max_iter=-1) #i risultati non convergevano, cosi' ho provato SVC piuttosto che LinearSVC

svm_optimized = GridSearchCV(svm_base, param_grid=params, cv=nfolds)
    #print ("Shape of X:", Xtr.shape)
svm_optimized.fit(Xtr, ytr)
    #print('Best params: {}'.format(svm_optimized.best_params_))

# evaluation of results
print('computing the cross-val score')
accuracies = cross_val_score(svm_optimized, Xtr, ytr, cv=nfolds, n_jobs=-1)
acc_mean, acc_std = accuracies.mean(), accuracies.std()
print('Accuracy={:.3f} (+-{:.3f})'.format(acc_mean, acc_std))

# final test
print('predicting the Epistolas')
    #ep1_ = svm_optimized.predict(ep1)
ep2= np.reshape(ep2, (1,-1))
ep2_ = svm_optimized.predict(ep2)
    #print('Epistola1 acc = {:.3f} {}'.format(ep1_.mean(), ep1_))
print('Epistola2 acc = {:.3f} {}'.format(ep2_.mean(), ep2_))

