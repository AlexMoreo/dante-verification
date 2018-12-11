from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator
from sklearn.svm import LinearSVC, SVC
from util.color_visualization import color

# DONE: ngrams should contain punctuation marks according to Sapkota et al. [39] in the PAN 2015 overview
# (More recently, it was shown that character
# n-grams corresponding to word affixes and including punctuation marks are the most
# significant features in cross-topic authorship attribution [57].)
# TODO: inspect the impact of chi-squared correlations against positive-only (or positive and negative) correlations for feature selection
# TODO: sentence length (Mendenhall-style)

for epistola in [1,2]:
    print('Epistola {}'.format(epistola))
    print('='*80)
    path = '../testi_{}'.format(epistola)

    positive, negative, ep_text = load_texts(path, unknown_target='EpistolaXIII_{}.txt'.format(epistola))

    feature_extractor = FeatureExtractor(function_words_freq='latin',
                                         conjugations_freq='latin',
                                         features_Mendenhall=True,
                                         tfidf=True, tfidf_feat_selection_ratio=0.1,
                                         wordngrams=(4,5),
                                         ngrams=True, ns=[4,5],
                                         split_documents=True,
                                         split_policy=split_by_sentences,
                                         window_size=3,
                                         normalize_features=True)

    Xtr,ytr = feature_extractor.fit_transform(positive, negative)
    ep, ep_fragments = feature_extractor.transform(ep_text, return_fragments=True, window_size=3)

    print('Fitting the Verificator')
    av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
    av.fit(Xtr,ytr)

    print('Predicting the Epistola {}'.format(epistola))
    title = 'Epistola {}'.format(epistola)
    av.predict(ep, title)
    fulldoc_prob, fragment_probs = av.predict_proba(ep, title)
    color(path='../dante_color/epistola{}.html'.format(epistola), texts=ep_fragments, probabilities=fragment_probs, title=title)

    param = 'All'
    # with open('features{}.csv'.format(epistola), 'at') as fo:
    #     validation=av.estimator.best_score_.mean()
    #     nfeatures = Xtr.shape[1]
    #     fo.write('{}\t{}\t{:.0f}\t{:.3f}\t{:.3f}\n'.format(param, value, nfeatures, validation, fulldoc_prob))
