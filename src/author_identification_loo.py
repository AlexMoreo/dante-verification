#import util._hide_sklearn_warnings
from data.dante_loader import load_latin_corpus, list_authors
from data.features import *
from model import AuthorshipVerificator, RangeFeatureSelector, leave_one_out
from util.evaluation import f1_from_counters
import argparse
from sklearn.pipeline import Pipeline

AUTHORS_CORPUS_I = ['Dante', 'ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba', 'PierDellaVigna']
AUTHORS_CORPUS_II = ['Dante', 'BeneFlorentinus', 'BenvenutoDaImola', 'BoncompagnoDaSigna', 'ClaraAssisiensis',
                           'FilippoVillani', 'GiovanniBoccaccio', 'GiovanniDelVirgilio', 'GrazioloBambaglioli', 'GuidoDaPisa',
                           'GuidoDeColumnis', 'GuidoFaba', 'IacobusDeVaragine', 'IohannesDeAppia', 'IohannesDePlanoCarpini',
                           'IulianusDeSpira', 'NicolaTrevet', 'PierDellaVigna', 'PietroAlighieri', 'RaimundusLullus',
                           'RyccardusDeSanctoGermano', 'ZonoDeMagnalis']


DEBUG_MODE = True


def main():
    log = open(args.log, 'wt')
    discarded = 0
    f1_scores = []
    counters = []
    for i, author in enumerate(args.authors):
        path = args.corpuspath
        print('='*80)
        print(f'Authorship Identification for {author} (complete {i}/{len(args.authors)})')
        print(f'Corpus {path}')
        print('-'*80)

        positive, negative, pos_files, neg_files, ep_text = load_latin_corpus(path, positive_author=author)
        files = np.asarray(pos_files + neg_files)
        if len(positive) < 2:
            discarded += 1
            print(f'discarding analysis for {author} which has only {len(positive)} documents')
            continue

        n_full_docs = len(positive) + len(negative)
        print(f'read {n_full_docs} documents from {path}')

        feature_extractor = FeatureExtractor(
            function_words_freq='latin',
            conjugations_freq='latin',
            features_Mendenhall=True,
            features_sentenceLengths=True,
            feature_selection_ratio=0.05 if DEBUG_MODE else 1,
            wordngrams=True, n_wordngrams=(1, 2),
            charngrams=True, n_charngrams=(3, 4, 5),
            preserve_punctuation=False,
            split_documents=True,
            split_policy=split_by_sentences,
            window_size=3,
            normalize_features=True
        )

        Xtr, ytr, groups = feature_extractor.fit_transform(positive, negative)

        print('Fitting the Verificator')
        #params = {'C': np.logspace(0, 1, 2)} if DEBUG_MODE else {'C': np.logspace(-3, +3, 7)}
        params = {'C': np.logspace(0, 1, 2)} if DEBUG_MODE else {'C': [1,10,100,1000,0.1,0.01,0.001]}

        slice_charngrams = feature_extractor.feature_range['_cngrams_task']
        slice_wordngrams = feature_extractor.feature_range['_wngrams_task']
        if slice_charngrams.start < slice_wordngrams.start:
            slice_first, slice_second = slice_charngrams, slice_wordngrams
        else:
            slice_first, slice_second = slice_wordngrams, slice_charngrams
        av = Pipeline([
            ('featsel_cngrams', RangeFeatureSelector(slice_second, 0.05)),
            ('featsel_wngrams', RangeFeatureSelector(slice_first, 0.05)),
            ('av', AuthorshipVerificator(C=1, param_grid=params))
        ])

        print('Validating the Verificator (Leave-One-Out)')
        score_ave, score_std, tp, fp, fn, tn = leave_one_out(
            av, Xtr, ytr, files, groups, test_lowest_index_only=True, counters=True
        )
        f1_scores.append(f1_from_counters(tp, fp, fn, tn))
        counters.append((tp, fp, fn, tn))
        tee(f'F1 for {author} = {f1_scores[-1]:.3f}', log)
        print(f'TP={tp} FP={fp} FN={fn} TN={tn}')

    print(f'Computing macro- and micro-averages (discarded {discarded}/{len(args.authors)})')
    f1_scores = np.array(f1_scores)
    counters = np.array(counters)

    macro_f1 = f1_scores.mean()
    micro_f1 = f1_from_counters(*counters.sum(axis=0).tolist())

    tee(f'LOO Macro-F1 = {macro_f1:.3f}', log)
    tee(f'LOO Micro-F1 = {micro_f1:.3f}', log)
    print()

    log.close()

    if DEBUG_MODE:
        print('DEBUG_MODE ON')


def tee(msg, log):
    print(msg)
    log.write(f'{msg}\n')
    log.flush()


if __name__ == '__main__':
    import os

    # Training settings
    parser = argparse.ArgumentParser(description='Authorship verification for MedLatin '
                                                 'submit each binary classifier to leave-one-out validation')
    parser.add_argument('corpuspath', type=str, metavar='CORPUSPATH',
                        help=f'Path to the directory containing the corpus (documents must be named '
                             f'<author>_<texname>.txt)')
    parser.add_argument('positive', type=str, default="Dante", metavar='AUTHOR',
                        help= f'Positive author for the hypothesis (default "Dante"); set to "ALL" to check '
                              f'every author')
    parser.add_argument('--log', type=str, metavar='PATH', default=None,
                        help='path to the log file where to write the results '
                             '(if not specified, then ./results_{corpuspath.name})')

    args = parser.parse_args()

    if args.positive == 'ALL':
        args.authors = list_authors(args.corpuspath, skip_prefix='Epistola')
    else:
        if (args.positive not in AUTHORS_CORPUS_I) and (args.positive in AUTHORS_CORPUS_II):
            print(f'warning: author {args.positive} is not in the known list of authors for CORPUS I nor CORPUS II')
        assert args.positive in list_authors(args.corpuspath, skip_prefix='Epistola'), 'unexpected author'
        args.authors = [args.positive]

    assert os.path.exists(args.corpuspath), f'corpus path {args.corpuspath} does not exist'

    main()

