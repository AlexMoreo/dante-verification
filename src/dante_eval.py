from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator
from sklearn.svm import LinearSVC, SVC
from util.color_visualization import color

# DONE: ngrams should contain punctuation marks according to Sapkota et al. [39] in the PAN 2015 overview
# (More recently, it was shown that character
# n-grams corresponding to word affixes and including punctuation marks are the most
# significant features in cross-topic authorship attribution [57].)  #we have cancelled the
# TODO: inspect the impact of chi-squared correlations against positive-only (or positive and negative) correlations for feature selection
# TODO: sentence length (Mendenhall-style) ?

for epistola in [1, 2]:
    print('Epistola {}'.format(epistola))
    print('='*80)
    path = '../testi_{}'.format(epistola)
    if epistola==2:
        path+='_with_GuidoDaPisa'

    positive, negative, ep_text = load_texts(path, unknown_target='EpistolaXIII_{}.txt'.format(epistola))
    n_full_docs = len(positive) + len(negative)

    feature_extractor = FeatureExtractor(function_words_freq='latin',
                                         conjugations_freq='latin',
                                         features_Mendenhall=True,
                                         tfidf_feat_selection_ratio=0.1,
                                         wordngrams=False, n_wordngrams=(1, 2),
                                         charngrams=True, n_charngrams=(3, 4, 5), preserve_punctuation=False,
                                         split_documents=True, split_policy=split_by_sentences, window_size=3,
                                         normalize_features=True)

    Xtr,ytr,groups = feature_extractor.fit_transform(positive, negative)
    print(ytr)

    ep, ep_fragments = feature_extractor.transform(ep_text, return_fragments=True, window_size=3)

    print('Fitting the Verificator')
    av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
    av.fit(Xtr,ytr,groups)

    print('Predicting the Epistola {}'.format(epistola))
    title = 'Epistola {}'.format('I' if epistola==1 else 'II')
    av.predict(ep, title)
    fulldoc_prob, fragment_probs = av.predict_proba(ep, title)
    # color(path='../dante_color/epistola{}.html'.format(epistola), texts=ep_fragments, probabilities=fragment_probs, title=title)

    score_ave, score_std = av.leave_one_out(Xtr, ytr, groups, test_lowest_index_only=False)
    print('LOO[full-and-fragments]={:.3f} +-{:.5f}'.format(score_ave, score_std))

    score_ave, score_std = av.leave_one_out(Xtr, ytr, groups, test_lowest_index_only=True)
    print('LOO[full-docs]={:.3f} +-{:.5f}'.format(score_ave, score_std))

    score_ave, score_std = av.leave_one_out(Xtr, ytr, None)
    print('LOO[w/o groups]={:.3f} +-{:.5f}'.format(score_ave, score_std))

