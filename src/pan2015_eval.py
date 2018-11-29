from joblib import Parallel
from joblib import delayed
from sklearn.linear_model import LogisticRegression
from util import disable_sklearn_warnings
from sklearn.svm import LinearSVC, SVC
from data.features import FeatureExtractor
from data.pan2015 import fetch_PAN2015, TaskGenerator
from model import AuthorshipVerificator
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def evaluation(y_pred, y_prob, y_true):
    y_pred_array = np.array(y_pred)
    y_prob_array = np.array(y_prob)
    y_true_array = np.array(y_true)

    acc = (y_pred_array == y_true_array).mean()
    f1 = f1_score(y_true_array, y_pred_array)
    auc = roc_auc_score(y_true_array, y_prob_array)
    pan_eval = acc * auc

    print('Accuracy = {:.3f}'.format(acc))
    print('F1 = {:.3f}'.format(f1))
    print('AUC = {:.3f}'.format(auc))
    print('Acc*AUC = {:.3f}'.format(pan_eval))
    print('true:', y_true)
    print('pred:', y_pred)

    return pan_eval


def doall(problem,pos,neg,test,truth):
    print('[Start]{}'.format(problem))
    feature_extractor = FeatureExtractor(function_words_freq=lang,
                                         features_Mendenhall=True,
                                         tfidf=False, tfidf_feat_selection_ratio=0.1,
                                         ngrams=True, ns=[4, 5],
                                         split_documents=False,
                                         normalize_features=True,
                                         verbose=True)

    method = AuthorshipVerificator(nfolds=3, estimator=LogisticRegression)

    X, y = feature_extractor.fit(pos, neg)
    test = feature_extractor.transform(test)

    method.fit(X, y)
    prediction = method.predict(test)
    if method.probability:
        probability = method.predict_proba(test)
    else:
        probability = prediction

    print('[End]{}'.format(problem))
    return problem, probability, prediction, truth

    # print('{}-->{:.3f} decision={}'.format(problem, probability, prediction))
    # print('pred={} truth={}'.format(prediction, truth))
    #
    # y_prob.append(probability)
    # y_pred.append(prediction)
    # y_true.append(truth)
    #
    # acc_auc = evaluation(y_pred, y_prob, y_true)



if __name__ == '__main__':
    split = 'test'
    lang = 'spanish'
    request = fetch_PAN2015(split, lang=lang)

    with open('results_ngrams.csv', 'wt') as fo:
        outcomes = Parallel(n_jobs=-1)(delayed(doall)(problem,pos,neg,test,truth) for problem,pos,neg,test,truth in TaskGenerator(request))
        y_pred, y_prob, y_true = [], [], []
        for problem, probability, prediction, truth in outcomes:
            fo.write('{} {:.3f}\n'.format(problem, probability))
            y_pred.append(prediction)
            y_prob.append(probability)
            y_true.append(truth)
        acc_auc = evaluation(y_pred, y_prob, y_true)
        print('ACC * AUC = {:.3f}'.format(acc_auc))


    print('done')