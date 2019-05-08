from sklearn.linear_model import LogisticRegression
from data.dante_loader import load_texts
from data.features import *
from model import AuthorshipVerificator, f1_from_counters
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_attribution(path, authors, attributions, paragraph_offset=1):

    paragraphs = ["Full"] + [f'{paragraph_offset+i}' for i in range(attributions.shape[1]-1)]

    fig, ax = plt.subplots()
    im = ax.imshow(attributions)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(paragraphs)))
    ax.set_yticks(np.arange(len(authors)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(paragraphs)
    ax.set_yticklabels(authors)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(authors)):
        for j in range(len(paragraphs)):
            text = ax.text(j, i, f'{attributions[i, j]:.2f}', ha="center", va="center", color="w")

    ax.set_title("Attribution matrix")
    fig.tight_layout()
    # plt.show()
    plt.savefig(path)

import sys
authors = ['Dante', 'ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba', 'PierDellaVigna']
attributions = np.load('attribution_ep1.npy')
plot_attribution('plot1.pdf', authors, attributions)
sys.exit(0)

author_attribution = []
for epistola in [1]:

    print(f'Epistola {epistola}')
    print('='*80)
    path = f'../testi_{epistola}'

    if epistola == 1:
        authors = ['Dante', 'ClaraAssisiensis', 'GiovanniBoccaccio', 'GuidoFaba', 'PierDellaVigna']
        paragraphs = range(1,3)

    else:
        authors = ['Dante', 'BeneFlorentinus', 'BenvenutoDaImola', 'BoncompagnoDaSigna', 'ClaraAssisiensis',
                   'FilippoVillani', 'GiovanniBoccaccio', 'GiovanniDelVirgilio',
                   'GrazioloBambaglioli', 'GuidoDaPisa',
                   'GuidoDeColumnis', 'GuidoFaba', 'IacobusDeVaragine', 'IohannesDeAppia',
                   'IohannesDePlanoCarpini', 'IulianusDeSpira', 'NicolaTrevet', 'PierDellaVigna',
                   'PietroAlighieri', 'RaimundusLullus',
                   'RyccardusDeSanctoGermano', 'ZonoDeMagnalis']
        paragraphs = range(13, 90)

    discarded = 0
    f1_scores = []
    counters = []
    for i, author in enumerate(authors):
        print('=' * 80)
        print('Authorship Identification for {} (complete {}/{})'.format(author, i, len(authors)))
        print('Corpus of Epistola {}'.format(epistola))
        print('=' * 80)

        target = [f'EpistolaXIII_{epistola}.txt'] + [f'EpistolaXIII_{epistola}_{paragraph}.txt' for paragraph in paragraphs]
        positive, negative, ep_texts = load_texts(path, positive_author=author, unknown_target=target)
        if len(positive) < 2:
            discarded += 1
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
    attribution_path = f'attribution_ep{epistola}.npy'
    print(f'saving attribution matrix of shape {author_attribution.shape} in {attribution_path}')
    np.save(attribution_path, author_attribution)






