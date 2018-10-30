import nltk
import numpy as np
import os
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.sparse import hstack, csr_matrix

function_words = ["et", "in", "de", "ad", "ut", "cum", "non", "per", "a", "que", "ex", "sed"]


# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
def _load_texts(path):
    # load the training data (all documents but Epistolas 1 and 2)
    documents = []
    authors   = []
    ndocs=0
    for file in os.listdir(path):
        if file.startswith('EpistolaXIII_'): continue
        file_clean = file.replace('.txt','')
        author, textname = file_clean.split('_')[0],file_clean.split('_')[1]
        text = open(join(path,file), encoding= "utf8").read()

        documents.append(text)
        authors.append(author)
        ndocs+=1

    # load the test data (Epistolas 1 and 2)
    ep1_text = open(join(path, 'EpistolaXIII_1.txt'), encoding="utf8").read()
    ep2_text = open(join(path, 'EpistolaXIII_2.txt'), encoding="utf8").read()

    return documents, authors, ep1_text, ep2_text



# ------------------------------------------------------------------------
# split policies
# ------------------------------------------------------------------------
# TODO: implement other split policies (e.g., overlapping ones, etc)
def _split_by_endline(text):
    return [t.strip() for t in text.split('\n') if t.strip()]


def splitter(documents, authors=None, split_policy=_split_by_endline):
    fragments = []
    authors_fragments = []
    for i, text in enumerate(documents):
        text_fragments = split_policy(text)
        fragments.extend(text_fragments)
        if authors is not None:
            authors_fragments.extend([authors[i]] * len(text_fragments))

    if authors is not None:
        return fragments, authors_fragments
    return fragments

# ------------------------------------------------------------------------
# feature extraction methods
# ------------------------------------------------------------------------
# TODO: implement other feature extraction methods
def _features_function_words_freq(documents):
    """
    Extract features as the frequency (x1000) of the functions used in the documents
    :param documents: a list where each element is the text (string) of a document
    :return: a np.array of shape (D,F) where D is len(documents) and F is len(function_words)
    """
    features = []
    for text in documents:
        tokens = nltk.word_tokenize(text)
        author_tokens = ([token.lower() for token in tokens if any(char.isalpha() for char in token)])
        freqs = nltk.FreqDist(author_tokens)

        nwords = len(author_tokens)
        funct_words_freq = [1000. * freqs[function_word] / nwords for function_word in function_words]

        features.append(funct_words_freq)

    return np.array(features)


def _features_tfidf(documents, tfidf_vectorizer=None):
    """
    Extract features as tfidf matrix extracted from the documents
    :param documents: a list where each element is the text (string) of a document
    :return: a tuple M,V, where M is an np.array of shape (D,F), with D being the len(documents) and F the number of
    distinct words; and V is the TfidfVectorizer already fit
    """
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)
        tfidf_vectorizer.fit(documents)

    features = tfidf_vectorizer.transform(documents)

    return features, tfidf_vectorizer

def _feature_selection(X, y, EP1, EP2, tfidf_feat_selection_ratio):
    nF = X.shape[1]
    num_feats = int(tfidf_feat_selection_ratio * nF)
    feature_selector = SelectKBest(chi2, k=num_feats)
    X = feature_selector.fit_transform(X, y)
    EP1 = feature_selector.transform(EP1)
    EP2 = feature_selector.transform(EP2)
    return X,EP1,EP2

def load_documents(path,
                   function_words_freq=True,
                   tfidf=False,
                   tfidf_feat_selection_ratio=1.,
                   split_documents=False,
                   split_policy = _split_by_endline,
                   verbose=True):
    """
    Loads the documents contained in path applying a number of feature extraction policies. The directory is assumed to
    contain files named according to <author>_<text_name>.txt plus two special files EpistolaXIII_1.txt and
    EpistolaXIII_2.txt concerning the two documents whose authorship attribution is to be determined.
    :param path: the path containing the texts, each named as <author>_<text_name>.txt
    :param function_words_freq: add the frequency of function words as features
    :param tfidf: add the tfidf as features
    :param split_documents: whether to split text into smaller documents or not (currenty, the policy is to split by '\n').
    Currently, the fragments resulting from the split are added to the pool of documents (i.e., they do not replace the
    full documents, which are anyway retained).
    :param split_policy: a callable that implements the split to be applied (ignored if split_documents=False)
    :param verbose: show information by stdout or not
    :return: np.arrays or csr_matrix (depending on whether tfidf is activated or not) X, y, EP1, EP2, where X is the
    matrix of features for the training set and y are the labels (np.array);
    EP1 and EP2 are the matrix of features for the epistola 1 (first row) and fragments (from row 2nd to last) if
    split_documents=True) and 2 (similar)
    """

    documents, authors, ep1_text, ep2_text = _load_texts(path)
    ep1,ep2 = [ep1_text],[ep2_text]
    n_original_docs=len(documents)

    if split_documents:
        doc_fragments, authors_fragments = splitter(documents, authors, split_policy=split_policy)
        documents.extend(doc_fragments)
        authors.extend(authors_fragments)

        ep1.extend(splitter(ep1, split_policy=split_policy))
        ep2.extend(splitter(ep2, split_policy=split_policy))


    # represent the target vector
    y = np.array([(1 if author == "Dante" else 0) for author in authors])

    # initialize the document-by-feature vector
    X = np.empty((len(documents), 0))
    EP1 = np.empty((len(ep1), 0))
    EP2 = np.empty((len(ep2), 0))

    if function_words_freq:
        X = np.hstack((X,_features_function_words_freq(documents)))
        EP1 = np.hstack((EP1, _features_function_words_freq(ep1)))
        EP2 = np.hstack((EP2, _features_function_words_freq(ep2)))

    if tfidf:
        X_features, vectorizer = _features_tfidf(documents)
        ep1_features, _ = _features_tfidf(ep1, vectorizer)
        ep2_features, _ = _features_tfidf(ep2, vectorizer)

        if tfidf_feat_selection_ratio < 1.:
            if verbose: print('feature selection')
            X_features, ep1_features, ep2_features = \
                _feature_selection(X_features, y, ep1_features, ep2_features, tfidf_feat_selection_ratio)

        # matrix is sparse now
        X   = hstack((csr_matrix(X), X_features))
        EP1 = hstack((csr_matrix(EP1), ep1_features))
        EP2 = hstack((csr_matrix(EP2), ep2_features))


    # print summary
    if verbose:
        print('load_documents: function_words_freq={} tfidf={}, split_documents={}, split_policy={}'
              .format(function_words_freq, tfidf, split_documents, split_policy.__name__))
        print('number of training (full) documents: {}'.format(n_original_docs))
        print('X shape (#documents,#features): {}'.format(X.shape))
        print('y prevalence: {:.2f}%'.format(y.mean()*100))
        print('Epistola 1 shape:', EP1.shape)
        print('Epistola 2 shape:', EP2.shape)
        print()

    return X, y, EP1, EP2
            
        
