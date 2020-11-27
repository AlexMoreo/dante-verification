from data.dante_loader import load_latin_corpus
from data.features import *
from model import AuthorshipVerificator
import settings
from util.evaluation import f1_from_counters, leave_one_out
import argparse
import pickle
import helpers
from helpers import tee
import os

def main():
    log = open(args.log, 'wt')
    discarded = 0
    f1_scores, acc_scores, counters = [], [], []
    path = args.corpuspath

    for i, author in enumerate(args.authors):

        print('='*80)
        print(f'[{args.corpus_name}] Authorship Identification for {author} (complete {i}/{len(args.authors)})')
        print('-'*80)

        pickle_file = f'Corpus{args.corpus_name}.Author{author}.window3.GFS1.pickle'
        if os.path.exists(pickle_file):
            print(f'pickle {pickle_file} exists... loading it')
            Xtr, ytr, groups, files, frange_chgrams, frange_wograms, fragments_range = pickle.load(open(pickle_file, 'rb'))
        else:
            print(f'pickle {pickle_file} noes not exists... generating it')
            positive, negative, pos_files, neg_files, ep_text = load_latin_corpus(path, positive_author=author)
            files = np.asarray(pos_files + neg_files)
            if len(positive) < 2:
                discarded += 1
                print(f'discarding analysis for {author} which has only {len(positive)} documents')
                continue

            n_full_docs = len(positive) + len(negative)
            print(f'read {n_full_docs} documents from {path}')

            feature_extractor = FeatureExtractor(**settings.config_feature_extraction)

            Xtr, ytr, groups = feature_extractor.fit_transform(positive, negative)
            frange_chgrams = feature_extractor.feature_range['_cngrams_task']
            frange_wograms = feature_extractor.feature_range['_wngrams_task']
            fragments_range = feature_extractor.fragments_range
            pickle.dump((Xtr, ytr, groups, files, frange_chgrams, frange_wograms, fragments_range),
                        open(pickle_file, 'wb'), pickle.HIGHEST_PROTOCOL)

        learner = args.learner.lower()
        av = AuthorshipVerificator(learner=learner, C=settings.DEFAULT_C, alpha=settings.DEFAULT_ALPHA,
                                   param_grid=settings.param_grid[learner], class_weight=args.class_weight,
                                   random_seed=settings.SEED, feat_selection_slices=[frange_chgrams, frange_wograms],
                                   feat_selection_ratio=args.featsel)

        print('Validating the Verificator (Leave-One-Out)')
        accuracy, f1, tp, fp, fn, tn, missclassified = leave_one_out(av, Xtr, ytr, files, groups)
        acc_scores.append(accuracy)
        f1_scores.append(f1)
        counters.append((tp, fp, fn, tn))

        tee(f'{author}',log)
        tee(f'\tF1 = {f1:.3f}', log)
        tee(f'\tAcc = {accuracy:.3f}', log)
        tee(f'\tTP={tp} FP={fp} FN={fn} TN={tn}', log)
        tee(f'\tErrors for {author}: {", ".join(missclassified)}', log)

    print(f'Computing macro- and micro-averages (discarded {discarded}/{len(args.authors)})')
    counters = np.array(counters)

    acc_mean = np.array(acc_scores).mean()
    macro_f1 = np.array(f1_scores).mean()
    micro_f1 = f1_from_counters(*counters.sum(axis=0).tolist())

    tee(f'LOO Macro-F1 = {macro_f1:.3f}', log)
    tee(f'LOO Micro-F1 = {micro_f1:.3f}', log)
    tee(f'LOO Accuracy = {acc_mean:.3f}', log)

    log.close()


if __name__ == '__main__':

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
                        help='path to the log file where to write the results (if not specified, then the name is'
                             'automatically generated from the arguments and stored in ../results/)')
    parser.add_argument('--featsel', default=0.1, metavar='FEAT_SEL_RATIO',
                        help=f'feature selection ratio for char- and word-ngrams')
    parser.add_argument('--class_weight', type=str, default=settings.CLASS_WEIGHT, metavar='CLASS_WEIGHT',
                        help=f"whether or not to reweight classes' importance")
    parser.add_argument('--learner', type=str, default='lr', metavar='LEARNER',
                        help=f"classification learner (lr, svm, mnb)")

    args = parser.parse_args()

    helpers.check_author(args)
    helpers.check_feat_sel_range(args)
    helpers.check_class_weight(args)
    helpers.check_corpus_path(args)
    helpers.check_learner(args)
    helpers.check_log_loo(args)

    main()

