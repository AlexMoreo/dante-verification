from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_latin_corpus, list_authors
from data.features import *
from model import AuthorshipVerificator
from util.evaluation import f1_from_counters
import argparse

AUTHORS_CORPUS_I = ['Dante', 'ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba', 'PierDellaVigna']
AUTHORS_CORPUS_II = ['Dante', 'BeneFlorentinus', 'BenvenutoDaImola', 'BoncompagnoDaSigna', 'ClaraAssisiensis',
                           'FilippoVillani', 'GiovanniBoccaccio', 'GiovanniDelVirgilio', 'GrazioloBambaglioli', 'GuidoDaPisa',
                           'GuidoDeColumnis', 'GuidoFaba', 'IacobusDeVaragine', 'IohannesDeAppia', 'IohannesDePlanoCarpini',
                           'IulianusDeSpira', 'NicolaTrevet', 'PierDellaVigna', 'PietroAlighieri', 'RaimundusLullus',
                           'RyccardusDeSanctoGermano', 'ZonoDeMagnalis']

def main():
    discarded = 0
    f1_scores = []
    counters = []
    for i, author in enumerate(args.authors):
        path = args.corpuspath
        print('='*80)
        print(f'Authorship Identification for {author} (complete {i}/{len(args.authors)})')
        print(f'Corpus {path}')
        print('='*80)

        positive, negative, pos_files, neg_files, ep_text = load_latin_corpus(
            path, positive_author=author, unknown_target=args.unknown
        )
        files = np.asarray(pos_files + neg_files)
        if len(positive) < 2:
            discarded += 1
            continue

        n_full_docs = len(positive) + len(negative)
        print(f'read {n_full_docs} documents from {path}')

        feature_extractor = FeatureExtractor(
            function_words_freq='latin',
            conjugations_freq='latin',
            features_Mendenhall=True,
            features_sentenceLengths=True,
            tfidf_feat_selection_ratio=0.1,
            wordngrams=True, n_wordngrams=(1, 2),
            charngrams=True, n_charngrams=(3, 4, 5),
            preserve_punctuation=False,
            split_documents=True,
            split_policy=split_by_sentences,
            window_size=3,
            normalize_features=True
        )

        Xtr, ytr, groups = feature_extractor.fit_transform(positive, negative)
        av = AuthorshipVerificator(nfolds=10, estimator=LogisticRegression)
        if args.unknown:
            print(f'Checking for the hypothesis that {author} was the author of {args.unknown}')
            ep, ep_fragments = feature_extractor.transform(ep_text, return_fragments=True, window_size=3)

            print('Fitting the Verificator')
            av.fit(Xtr, ytr, groups)
            av.predict_proba(ep, args.unknown)

        if args.loo:
            print('Validating the Verificator (Leave-One-Out)')
            score_ave, score_std, tp, fp, fn, tn = av.leave_one_out(
                Xtr, ytr, files, groups, test_lowest_index_only=True, counters=True
            )
            f1_scores.append(f1_from_counters(tp, fp, fn, tn))
            counters.append((tp, fp, fn, tn))
            print(f'F1 for {author} = {f1_scores[-1]:.3f}')

    if args.loo:
        print(f'Computing macro- and micro-averages (discarded {discarded}/{len(args.authors)})')
        f1_scores = np.array(f1_scores)
        counters = np.array(counters)

        macro_f1 = f1_scores.mean()
        micro_f1 = f1_from_counters(*counters.sum(axis=0).tolist())

        print(f'Macro-F1 = {macro_f1:.3f}')
        print(f'Micro-F1 = {micro_f1:.3f}')
        print()


if __name__ == '__main__':
    import os

    # Training settings
    parser = argparse.ArgumentParser(description='Authorship verification for Epistola XIII')
    parser.add_argument('corpuspath', type=str, metavar='PATH',
                        help=f'Path to the directory containing the corpus (documents must be named <author>_<texname>.txt')
    parser.add_argument('positive', type=str, default="Dante",
                        help= f'Positive author for the hypothesis (default "Dante"); set to "ALL" to check every author')
    parser.add_argument('--loo', default=False, action='store_true',
                        help='submit each binary classifier to leave-one-out validation')
    parser.add_argument('--unknown', type=str, default=None,
                        help='path to the file of unknown paternity (default None)')

    args = parser.parse_args()

    if args.positive == 'ALL':
        args.authors = list_authors(args.corpuspath, skip_prefix='Epistola')
    else:
        if (args.positive not in AUTHORS_CORPUS_I) and (args.positive in AUTHORS_CORPUS_II):
            print(f'warning: author {args.positive} is not in the known list of authors for CORPUS I nor CORPUS II')
        assert args.positive in list_authors(args.corpuspath, skip_prefix='Epistola'), 'unexpected author'
        args.authors = [args.positive]

    assert args.unknown or args.loo, 'error: nor an unknown document, nor LOO have been requested. Nothing to do.'
    assert args.unknown is None or os.path.exists(args.unknown), 'unknown file does not exist'

    main()