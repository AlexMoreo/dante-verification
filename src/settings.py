import numpy as np
from data.features import split_by_sentences

AUTHORS_CORPUS_I = [
    'Dante',
    'ClaraAssisiensis',
    'GiovanniBoccaccio',
    'GuidoFaba',
    'PierDellaVigna'
]

AUTHORS_CORPUS_II = [
    'Dante',
    'BeneFlorentinus',
    'BenvenutoDaImola',
    'BoncompagnoDaSigna',
    'ClaraAssisiensis',
    'FilippoVillani',
    'GiovanniBoccaccio',
    'GiovanniDelVirgilio',
    'GrazioloBambaglioli',
    'GuidoDaPisa',
    'GuidoDeColumnis',
    'GuidoFaba',
    'IacobusDeVaragine',
    'IohannesDeAppia',
    'IohannesDePlanoCarpini',
    'IulianusDeSpira',
    'NicolaTrevet',
    'PierDellaVigna',
    'PietroAlighieri',
    'RaimundusLullus',
    'RyccardusDeSanctoGermano',
    'ZonoDeMagnalis'
]

DEFAULT_C = 0.1
DEFAULT_ALPHA = 0.001
CLASS_WEIGHT = 'balanced'
SEED = 1

grid_C = np.logspace(-3,3,7)
param_grid = {
    'lr': {'C': grid_C},
    'svm': {'C': grid_C},
    'mnb': {'alpha': np.logspace(-7,-1,7)}
}

config_feature_extraction = {
    'function_words_freq': 'latin',
    'conjugations_freq': 'latin',
    'features_Mendenhall': True,
    'features_sentenceLengths': True,
    'feature_selection_ratio': 1,
    'wordngrams': True,
    'n_wordngrams': (1, 2),
    'charngrams': True,
    'n_charngrams': (3, 4, 5),
    'preserve_punctuation': False,
    'split_documents': True,
    'split_policy': split_by_sentences,
    'window_size': 3,
    'normalize_features': True
}
