from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator, f1_from_counters
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_attribution(path, authors, attributions, paragraph_offset=1, figsize=(5,5), label_offset=0.3):

    attributions = attributions.T
    print(attributions.shape)
    # attributions=attributions>0.5
    paragraphs = ["Full"] + [f'{paragraph_offset+i}' for i in range(attributions.shape[0]-1)]

    fig, ax = plt.subplots(figsize=figsize)

    # im = ax.imshow(attributions, vmin=0, vmax=1, cmap='Greens')
    im = ax.imshow(attributions, vmin=0, vmax=1, cmap='Greys')

    # Create colorbar
    # cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1, pad=0.04)
    # ax.figure.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1)
    # ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)

    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    # ax.set_xticks(np.arange(len(authors)))
    ax.set_xticks(np.arange(len(authors) + 0) + label_offset)
    ax.set_yticks(np.arange(len(paragraphs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(authors)
    ax.set_yticklabels(paragraphs)

    ax.tick_params(top=False, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(authors)+1) - .5, minor=True)
    ax.set_yticks(np.arange(len(paragraphs)+1) - .5, minor=True)

    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    # for i in range(len(authors)):
    #     for j in range(len(paragraphs)):
    #         text = ax.text(j, i, f'{attributions[i, j]:.2f}', ha="center", va="center", color="w")

    # ax.set_title("Attribution matrix")
    fig.tight_layout()
    # plt.show()
    plt.savefig(path)

import sys
authors1 = ['ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba', 'PierDellaVigna']
authors2 = ['BeneFlorentinus', 'BenvenutoDaImola', 'BoncompagnoDaSigna',
                   'FilippoVillani', 'GiovanniBoccaccio', 'GiovanniDelVirgilio',
                   'GrazioloBambaglioli', 'GuidoDaPisa',
                   'GuidoDeColumnis', 'GuidoFaba', 'IacobusDeVaragine', 'IohannesDeAppia',
                   'IohannesDePlanoCarpini', 'IulianusDeSpira', 'NicolaTrevet',
                   'PietroAlighieri', 'RaimundusLullus',
                   'RyccardusDeSanctoGermano', 'ZonoDeMagnalis']
authors3 = sorted(np.unique(authors1 + authors2).tolist())

for epistola in [1]:
    paragraph_offset = 1
    label_offset = 0.2
    if epistola == 1:
        authors = ['Dante'] + authors1
        figsize = (4, 4)
    elif epistola == 2:
        authors = ['Dante'] + authors2
        figsize = (6, 4)
    else:
        authors = ['Dante'] + authors3

    attributions = np.load(f'attribution_ep{epistola}_xiv.npy')
    plot_attribution(f'plot{epistola}_xiv.png', authors, attributions, paragraph_offset=paragraph_offset, figsize=figsize, label_offset=label_offset)
sys.exit(0)

for epistola in [1]:

    author_attribution = []
    print(f'Epistola {epistola}')
    print('='*80)
    path = f'../testiXIV_{epistola}'


    if epistola == 1:
        authors = ['Dante'] + authors1
    elif epistola == 2:
        authors = ['Dante'] + authors2
    else:
        authors = ['Dante'] + authors3

    discarded = 0
    f1_scores = []
    counters = []
    for i, author in enumerate(authors):
        print('=' * 80)
        print('Authorship Identification for {} (complete {}/{})'.format(author, i, len(authors)))
        print('Corpus of Epistola {}'.format(epistola))
        print('=' * 80)

        target = [f'Epistola_ArigoVII.txt'] + [f'Epistola_ArigoVII_{paragraph}.txt' for paragraph in range(1,6)]
        positive, negative, ep_texts = load_texts(path, positive_author=author, unknown_target=target, train_skip_prefix='Epistola_ArigoVII')

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

        attributions=[]
        for i,target_text in enumerate(ep_texts):
            ep = feature_extractor.transform(target_text, avoid_splitting=True)
            prob,_ = av.predict_proba(ep, epistola_name=target[i])
            attributions.append(prob)
        author_attribution.append(attributions)

    author_attribution = np.asarray(author_attribution)
    attribution_path = f'attribution_ep{epistola}_xiv.npy'
    print(f'saving attribution matrix of shape {author_attribution.shape} in {attribution_path}')
    np.save(attribution_path, author_attribution)






