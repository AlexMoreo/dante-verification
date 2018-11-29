from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator
from sklearn.svm import LinearSVC, SVC

# DONE: ngrams should contain punctuation marks according to Sapkota et al. [39] in the PAN 2015 overview
# (More recently, it was shown that character
# n-grams corresponding to word affixes and including punctuation marks are the most
# significant features in cross-topic authorship attribution [57].)
# TODO: split policies: understand overlapping in cross-validation



path = '../testi'

positive, negative, ep1_text, ep2_text = load_texts(path)

feature_extractor = FeatureExtractor(function_words_freq='latin', features_Mendenhall=True,
                                     tfidf=False, tfidf_feat_selection_ratio=0.1,
                                     wordngrams=(4,5),
                                     ngrams=True, ns=[4,5],
                                     split_documents=True,
                                     split_policy=split_by_sentences,
                                     window_size=3,
                                     normalize_features=True,  verbose=True)

Xtr,ytr = feature_extractor.fit(positive, negative)
ep1 = feature_extractor.transform(ep1_text)
ep2 = feature_extractor.transform(ep2_text)

print('Fitting the Verificator')
av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
av.fit(Xtr,ytr)

print('Predicting the Epistolas')
av.predict(ep1, 'Epistola 1')
av.predict_proba(ep1, 'Epistola 1')

av.predict(ep2, 'Epistola 2')
av.predict_proba(ep2, 'Epistola 2')
