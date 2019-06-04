from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator, f1_from_counters
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

for epistola in [2]:

    author_attribution = []
    print(f'Epistola {epistola}')
    print('='*80)
    path = f'../testi_{epistola}'
    if epistola==2: path+='_tutti'

    author = 'Dante'
    print('=' * 80)
    print('Corpus of Epistola {}'.format(epistola))
    print('=' * 80)

    positive, negative, ep_texts = load_texts(path, positive_author=author, unknown_target=f'EpistolaXIII_{epistola}.txt')

    n_full_docs = len(positive) + len(negative)

    feature_extractor = FeatureExtractor(function_words_freq='latin',
                                         conjugations_freq='latin',
                                         features_Mendenhall=True,
                                         features_sentenceLengths=True,
                                         tfidf_feat_selection_ratio=0.1,
                                         wordngrams=True, n_wordngrams=(1, 2),
                                         charngrams=True, n_charngrams=(3, 4, 5),
                                         preserve_punctuation=False,
                                         split_documents=True, split_policy=split_by_sentences, window_size=3,
                                         normalize_features=True)

    Xtr, ytr, groups = feature_extractor.fit_transform(positive, negative)

    print('Fitting the Verificator')
    av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression, author_name=author)
    av.fit(Xtr, ytr, groups)

    feat_rank = np.argsort(av.estimator.coef_[0])
    coef_ordered = av.estimator.coef_[0][feat_rank]
    feat_name_ordered = feature_extractor.feature_names[feat_rank]

    print('Most Dantesque features::')
    for i in range(100):
        print(f'{i}: {feat_name_ordered[::-1][i]} {coef_ordered[::-1][i]:.3f}')

    print('\nMost Non-Dantesque features::')
    for i in range(100):
        print(f'{i}: {feat_name_ordered[i]} {coef_ordered[i]:.3f}')


    print('done')



