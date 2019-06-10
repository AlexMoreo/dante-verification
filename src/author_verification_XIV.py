from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator, f1_from_counters
from sklearn.svm import LinearSVC, SVC
from util.color_visualization import color
import pickle
import os

for epistola in [1,2,3]: #3 means "both Ep1 and Ep2 corpora"

    print('Epistola {}'.format(epistola))
    print('='*80)
    path = '../testiXIV_{}'.format(epistola)
    paragraphs = range(1, 6)

    target = [f'Epistola_ArigoVII.txt'] + [f'Epistola_ArigoVII_{paragraph}.txt' for paragraph in paragraphs]
    positive, negative, _, _, ep_texts = load_texts(path, positive_author='Dante', unknown_target=target, train_skip_prefix='Epistola_ArigoVII')

    pickle_file = f'../dante_color/epistola{epistola}_xiv.pkl'
    if os.path.exists(pickle_file):
        print(f'loading pickle file {pickle_file}')
        probabilities = pickle.load(open(pickle_file, 'rb'))
    else:
        print(f'generating pickle file')
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

        Xtr,ytr,groups = feature_extractor.fit_transform(positive, negative)
        print(ytr)

        print('Fitting the Verificator')
        av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression, author_name='Dante')
        av.fit(Xtr,ytr,groups)

        probabilities = []
        for i, target_text in enumerate(ep_texts):
            ep = feature_extractor.transform(target_text, avoid_splitting=True)
            prob, _ = av.predict_proba(ep, epistola_name=target[i])
            probabilities.append(prob)

        pickle.dump(probabilities, open(pickle_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    color(path=f'../dante_color/epistola{epistola}_xiv.html', texts=ep_texts,
          probabilities=probabilities, title=f'Epistola {epistola}',
          paragraph_offset=paragraphs[0])


    # print('Predicting the Epistola {}'.format(epistola))
    # title = 'Epistola {}'.format('I' if epistola==1 else 'II')
    # av.predict(ep, title)
    # fulldoc_prob, fragment_probs = av.predict_proba(ep, title)
    # color(path='../dante_color/epistola{}.html'.format(epistola), texts=ep_fragments, probabilities=fragment_probs, title=title)

    # score_ave, score_std = av.leave_one_out(Xtr, ytr, groups, test_lowest_index_only=False)
    # print('LOO[full-and-fragments]={:.3f} +-{:.5f}'.format(score_ave, score_std))

    # score_ave, score_std, tp, fp, fn, tn = av.leave_one_out(Xtr, ytr, groups, test_lowest_index_only=True, counters=True)
    # print('LOO[full-docs]={:.3f} +-{:.5f}'.format(score_ave, score_std))
    # f1_ = f1_from_counters(tp, fp, fn, tn)
    # print('F1 = {:.3f}'.format(f1_))

    # score_ave, score_std = av.leave_one_out(Xtr, ytr, None)
    # print('LOO[w/o groups]={:.3f} +-{:.5f}'.format(score_ave, score_std))

