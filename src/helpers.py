import settings
from data.dante_loader import list_authors
import os
import pathlib


def tee(msg, log):
    print(msg)
    log.write(f'{msg}\n')
    log.flush()


def check_author(args):
    if args.positive == 'ALL':
        args.authors = list_authors(args.corpuspath, skip_prefix='Epistola')
    else:
        if (args.positive not in settings.AUTHORS_CORPUS_I) and (args.positive in settings.AUTHORS_CORPUS_II):
            print(f'warning: author {args.positive} is not in the known list of authors for CORPUS I nor CORPUS II')
        assert args.positive in list_authors(args.corpuspath, skip_prefix='Epistola'), 'unexpected author'
        args.authors = [args.positive]


def check_feat_sel_range(args):
    if not isinstance(args.featsel, float):
        if isinstance(args.featsel, str) and '.' in args.featsel:
            args.featsel = float(args.featsel)
        else:
            args.featsel = int(args.featsel)
    if isinstance(args.featsel, float):
        assert 0 < args.featsel <= 1, 'feature selection ratio out of range'


def check_class_weight(args):
    assert args.class_weight in ['balanced', 'none', 'None']
    if args.class_weight.lower() == 'none':
        args.class_weight = None


def check_corpus_path(args):
    assert os.path.exists(args.corpuspath), f'corpus path {args.corpuspath} does not exist'
    args.corpus_name = pathlib.Path(args.corpuspath).name


def check_learner(args):
    assert args.learner.lower() in settings.param_grid.keys(), \
        f'unknown learner, use any in {settings.param_grid.keys()}'


def check_log_loo(args):
    if args.log is None:
        os.makedirs('../results', exist_ok=True)
        args.log = f'../results/LOO_Corpus{args.corpus_name}.Author{args.positive}.' \
                   f'fs{args.featsel}.classweight{str(args.class_weight)}.CLS{args.learner}.txt'


def check_log_unknown(args):
    args.unknown_name = pathlib.Path(args.unknown).name
    if args.log is None:
        os.makedirs('../results', exist_ok=True)
        assert os.path.exists(args.unknown), f'file {args.unknown} does not exist'
        args.log = f'../results/Unknown{args.unknown_name}_Corpus{args.corpus_name}.Author{args.positive}.' \
                   f'fs{args.featsel}.classweight{str(args.class_weight)}.CLS{args.learner}.txt'