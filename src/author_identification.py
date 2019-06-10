from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator, f1_from_counters
from sklearn.svm import LinearSVC, SVC
from util.color_visualization import color

# DONE: ngrams should contain punctuation marks according to Sapkota et al. [39] in the PAN 2015 overview
# (More recently, it was shown that character
# n-grams corresponding to word affixes and including punctuation marks are the most
# significant features in cross-topic authorship attribution [57].)  #we have cancelled the
# TODO: inspect the impact of chi-squared correlations against positive-only (or positive and negative) correlations for feature selection
# TODO: sentence length (Mendenhall-style) ?


for epistola in [1]:
    if epistola==1:
        authors = ['Dante','ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba','PierDellaVigna']
    else:
        authors = ['Dante', 'BeneFlorentinus','BenvenutoDaImola', 'BoncompagnoDaSigna', 'ClaraAssisiensis',
                   'FilippoVillani', 'GiovanniBoccaccio', 'GiovanniDelVirgilio',
                   'GrazioloBambaglioli', 'GuidoDaPisa',
                   'GuidoDeColumnis', 'GuidoFaba','IacobusDeVaragine','IohannesDeAppia',
                   'IohannesDePlanoCarpini','IulianusDeSpira', 'NicolaTrevet', 'PierDellaVigna',
                   'PietroAlighieri', 'RaimundusLullus',
                   'RyccardusDeSanctoGermano','ZonoDeMagnalis']

    discarded = 0
    f1_scores = []
    counters = []
    for i,author in enumerate(authors):
        print('='*80)
        print('Authorship Identification for {} (complete {}/{})'.format(author, i, len(authors)))
        print('Corpus of Epistola {}'.format(epistola))
        print('='*80)
        path = '../testi_{}'.format(epistola)
        if epistola==2:
            path+='_interaEpistola'

        positive, negative, pos_files, neg_files, ep_text = load_texts(path, positive_author=author, unknown_target='EpistolaXIII_{}.txt'.format(epistola))
        files = np.asarray(pos_files + neg_files)
        if len(positive) < 2:
            discarded+=1
            continue

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

        ep, ep_fragments = feature_extractor.transform(ep_text, return_fragments=True, window_size=3)

        print('Fitting the Verificator')
        av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
        av.fit(Xtr,ytr,groups)

        score_ave, score_std, tp, fp, fn, tn = av.leave_one_out(Xtr, ytr, files, groups, test_lowest_index_only=True, counters=True)
        # print('LOO[full-docs]={:.3f} +-{:.5f}'.format(score_ave, score_std))
        f1_scores.append(f1_from_counters(tp, fp, fn, tn))
        counters.append((tp, fp, fn, tn))
        print('F1 for {} = {:.3f}'.format(author,f1_scores[-1]))


    print('Computing macro- and micro-averages (discarded {}/{})'.format(discarded,len(authors)))
    f1_scores = np.array(f1_scores)
    counters = np.array(counters)

    macro_f1 = f1_scores.mean()
    micro_f1 = f1_from_counters(*counters.sum(axis=0).tolist())

    print('Macro-F1 = {:.3f}'.format(macro_f1))
    print('Micro-F1 = {:.3f}'.format(micro_f1))
    print()


