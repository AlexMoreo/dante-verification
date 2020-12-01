from data.dante_loader import load_latin_corpus, list_authors
from data.features import *
from model import AuthorshipVerificator
import settings
import argparse
import helpers
from helpers import tee
import os
import pickle


def main():
    log = open(args.log, 'wt')

    discarded = 0
    path = args.corpuspath

    for i, author in enumerate(args.authors):
        print('='*80)
        print(f'[{args.corpus_name}] Authorship Identification for {author} (complete {i}/{len(args.authors)})')
        print('-'*80)

        pickle_file = f'Corpus{args.corpus_name}.Author{author}.window3.GFS{args.featsel}.unk{args.unknown_name}.pickle'
        if os.path.exists(pickle_file):
            print(f'pickle {pickle_file} exists... loading it')
            Xtr, ytr, groups, files, frange_chgrams, frange_wograms, fragments_range, ep, ep_fragments = \
                pickle.load(open(pickle_file, 'rb'))
        else:
            print(f'pickle {pickle_file} noes not exists... generating it')
            positive, negative, pos_files, neg_files, ep_text = \
                load_latin_corpus(path, positive_author=author, unknown_target=args.unknown)
            files = np.asarray(pos_files + neg_files)
            if len(positive) < 2:
                discarded += 1
                print(f'discarding analysis for {author} which has only {len(positive)} documents')
                continue

            n_full_docs = len(positive) + len(negative)
            print(f'read {n_full_docs} documents from {path}')

            settings.config_feature_extraction['feature_selection_ratio'] = args.featsel
            feature_extractor = FeatureExtractor(**settings.config_feature_extraction)

            Xtr, ytr, groups = feature_extractor.fit_transform(positive, negative)
            frange_chgrams = feature_extractor.feature_range['_cngrams_task']
            frange_wograms = feature_extractor.feature_range['_wngrams_task']
            fragments_range = feature_extractor.fragments_range

            ep, ep_fragments = feature_extractor.transform(ep_text, return_fragments=True, window_size=3)

            pickle.dump((Xtr, ytr, groups, files, frange_chgrams, frange_wograms, fragments_range, ep, ep_fragments),
                        open(pickle_file, 'wb'), pickle.HIGHEST_PROTOCOL)

        learner = args.learner.lower()
        av = AuthorshipVerificator(learner=learner, C=settings.DEFAULT_C, alpha=settings.DEFAULT_ALPHA,
                                   param_grid=settings.param_grid[learner], class_weight=args.class_weight,
                                   random_seed=settings.SEED)

        av.fit(Xtr, ytr, groups)

        print(f'Checking for the hypothesis that {author} was the author of {args.unknown_name}')
        pred = av.predict_proba(ep)
        pred = pred[0,1]
        tee(f'{args.unknown}: Posterior probability for {author} is {pred:.4f};\n'
            f'this means the classifier attributes the text to {"not-" if pred<0.5 else ""}{author}', log)

    log.close()


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Authorship verification for a text of unknown paternity')
    parser.add_argument('corpuspath', type=str, metavar='CORPUSPATH',
                        help=f'Path to the directory containing the corpus (documents must be named '
                             f'<author>_<texname>.txt)')
    parser.add_argument('positive', type=str, default="Dante", metavar='AUTHOR',
                        help= f'Positive author for the hypothesis (default "Dante"); set to "ALL" to check '
                              f'every author')
    parser.add_argument('unknown', type=str, metavar='PATH', default=None,
                        help='path to the file of unknown paternity (default None)')
    parser.add_argument('--log', type=str, metavar='PATH', default=None,
                        help='path to the log file where to write the results (if not specified, then the name is'
                             'automatically generated from the arguments and stored in ../results/)')
    parser.add_argument('--featsel', default=0.1, metavar='FEAT_SEL_RATIO',
                        help=f'feature selection ratio for char- and word-ngrams')
    parser.add_argument('--class_weight', type=str, default=settings.CLASS_WEIGHT, metavar='CLASS_WEIGHT',
                        help=f"whether or not to reweight classes' importance")
    parser.add_argument('--learner', type=str, default='LR', metavar='LEARNER',
                        help=f"classification learner (LR, SVM, MNB, RF)")

    args = parser.parse_args()

    helpers.check_author(args)
    helpers.check_feat_sel_range(args)
    helpers.check_class_weight(args)
    helpers.check_corpus_path(args)
    helpers.check_learner(args)
    helpers.check_log_unknown(args)

    main()

