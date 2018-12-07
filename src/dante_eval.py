from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator
from sklearn.svm import LinearSVC, SVC

# DONE: ngrams should contain punctuation marks according to Sapkota et al. [39] in the PAN 2015 overview
# (More recently, it was shown that character
# n-grams corresponding to word affixes and including punctuation marks are the most
# significant features in cross-topic authorship attribution [57].)
# TODO: inspect the impact of chi-squared correlations against positive-only (or positive and negative) correlations for feature selection
from util.color_visualization import color

path = '../testi_2'

positive, negative, ep1_text, ep2_text = load_texts(path)

feature_extractor = FeatureExtractor(function_words_freq='latin',
                                     conjugations_freq='latin',
                                     features_Mendenhall=True,
                                     tfidf=False, tfidf_feat_selection_ratio=0.1,
                                     wordngrams=(4,5),
                                     ngrams=True, ns=[4,5],
                                     split_documents=True,
                                     split_policy=split_by_sentences,
                                     window_size=3,
                                     normalize_features=True,
                                     verbose=True)

Xtr,ytr = feature_extractor.fit(positive, negative)
ep1,ep1_fragments = feature_extractor.transform(ep1_text, return_fragments=True, window_size=3)
ep2,ep2_fragments = feature_extractor.transform(ep2_text, return_fragments=True, window_size=3)

print('Fitting the Verificator')
av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
av.fit(Xtr,ytr)

print('Predicting the Epistolas')
# av.predict(ep1, 'Epistola 1')
# fulldoc_prob,fragment_probs = av.predict_proba(ep1, 'Epistola 1')
# color(path='../dante_color/epistola1.html', texts=ep1_fragments, probabilities=fragment_probs, title='Epistola 1')

av.predict(ep2, 'Epistola 2')
fulldoc_prob,fragment_probs = av.predict_proba(ep2, 'Epistola 2')
color(path='../dante_color/epistola2.html', texts=ep2_fragments, probabilities=fragment_probs, title='Epistola 2')
