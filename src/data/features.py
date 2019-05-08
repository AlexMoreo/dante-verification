import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix, issparse
from nltk.corpus import stopwords


latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']

latin_conjugations = ['o', 'eo', 'io', 'as', 'es', 'is', 'at', 'et', 'it', 'amus', 'emus', 'imus', 'atis', 'etis',
                      'itis', 'ant', 'ent', 'unt', 'iunt', 'or', 'eor', 'ior', 'aris', 'eris', 'iris', 'atur', 'etur',
                      'itur', 'amur', 'emur', 'imur', 'amini', 'emini', 'imini', 'antur', 'entur', 'untur', 'iuntur',
                      'abam', 'ebam', 'iebam',  'abas', 'ebas', 'iebas', 'abat', 'ebat', 'iebat', 'abamus', 'ebamus',
                      'iebamus', 'abatis', 'ebatis', 'iebatis', 'abant', 'ebant', 'iebant', 'abar', 'ebar', 'iebar',
                      'abaris', 'ebaris', 'iebaris', 'abatur', 'ebatur', 'iebatur', 'abamur', 'ebamur', 'iebamur',
                      'abamini', 'ebamini', 'iebamini', 'abantur', 'ebantur', 'iebantur', 'abo', 'ebo', 'am', 'iam',
                      'abis', 'ebis', 'ies', 'abit', 'ebit', 'iet', 'abimus', 'ebimus', 'emus', 'iemus', 'abitis',
                      'ebitis', 'ietis', 'abunt', 'ebunt', 'ient', 'abor', 'ebor', 'ar', 'iar', 'aberis', 'eberis',
                      'ieris', 'abitur', 'ebitur', 'ietur', 'abimur', 'ebimur', 'iemur', 'abimini', 'ebimini', 'iemini',
                      'abuntur', 'ebuntur', 'ientur', 'i', 'isti', 'it', 'imus', 'istis', 'erunt', 'em', 'eam', 'eas',
                      'ias', 'eat', 'iat', 'eamus', 'iamus', 'eatis', 'iatis', 'eant', 'iant', 'er', 'ear', 'earis',
                      'iaris', 'eatur', 'iatur', 'eamur', 'iamur', 'eamini', 'iamini', 'eantur', 'iantur', 'rem', 'res',
                      'ret', 'remus', 'retis', 'rent', 'rer', 'reris', 'retur', 'remur', 'remini', 'rentur', 'erim',
                      'issem', 'isses', 'isset', 'issemus', 'issetis', 'issent', 'a', 'ate', 'e', 'ete', 'ite', 'are',
                      'ere', 'ire', 'ato', 'eto', 'ito', 'atote', 'etote', 'itote', 'anto', 'ento', 'unto', 'iunto',
                      'ator', 'etor', 'itor', 'aminor', 'eminor', 'iminor', 'antor', 'entor', 'untor', 'iuntor', 'ari',
                      'eri', 'iri', 'andi', 'ando', 'andum', 'andus', 'ande', 'ans', 'antis', 'anti', 'antem', 'antes',
                      'antium', 'antibus', 'antia', 'esse', 'sum', 'es', 'est', 'sumus', 'estis', 'sunt', 'eram', 'eras',
                      'erat', 'eramus', 'eratis', 'erant', 'ero', 'eris', 'erit', 'erimus', 'eritis', 'erint', 'sim',
                      'sis', 'sit', 'simus', 'sitis', 'sint', 'essem', 'esses', 'esset', 'essemus', 'essetis', 'essent',
                      'fui', 'fuisti', 'fuit', 'fuimus', 'fuistis', 'fuerunt', 'este', 'esto', 'estote', 'sunto']

spanish_conjugations = ['o','as','a','amos','áis','an','es','e','emos','éis','en','imos','ís','guir','ger','gir',
                        'ar', 'er', 'ir', 'é', 'aste', 'ó','asteis','aron','í','iste','ió','isteis','ieron',
                        'aba', 'abas', 'ábamos', 'aban', 'ía', 'ías', 'íamos', 'íais', 'ían', 'ás','á',
                        'án','estoy','estás','está','estamos','estáis','están']


def get_function_words(lang):
    if lang=='latin':
        return latin_function_words
    elif lang in ['english','spanish']:
        return stopwords.words(lang)
    else:
        raise ValueError('{} not in scope!'.format(lang))

def get_conjugations(lang):
    if lang == 'latin':
        return latin_conjugations
    elif lang == 'spanish':
        return spanish_conjugations
    else:
        raise ValueError('conjugations for languages other than Latin and Spanish are not yet supported')


# ------------------------------------------------------------------------
# split policies
# ------------------------------------------------------------------------
# TODO: implement other split policies (e.g., overlapping ones, etc)
def split_by_endline(text):
    return [t.strip() for t in text.split('\n') if t.strip()]


def split_by_sentences(text):
    sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
    #sentences= [t.strip() for t in re.split(r"\.|\?|\!\;", text) if t.strip()]

    for i,sentence in enumerate(sentences):
        unmod_tokens = nltk.tokenize.word_tokenize(sentence)
        mod_tokens = ([token for token in unmod_tokens if any(char.isalpha() for char in token)])
        if len(mod_tokens)<8:
            if i<len(sentences)-1:
                sentences[i+1] = sentences[i] + ' ' + sentences[i+1]
            else:
                sentences[i-1] = sentences[i-1] + ' ' + sentences[i]
            sentences.pop(i)

    return sentences

def windows(text_fragments, window_size):
    new_fragments = []
    nbatches = len(text_fragments) // window_size
    if len(text_fragments) % window_size > 0:
        nbatches+=1
    # for i in range(len(text_fragments)-window_size+1):
    for i in range(nbatches):
        offset = i*window_size
        new_fragments.append(' '.join(text_fragments[offset:offset+window_size]))
    return new_fragments

def splitter(documents, authors=None, split_policy=split_by_sentences, window_size=1):
    fragments = []
    authors_fragments = []
    groups = []
    for i, text in enumerate(documents):
        text_fragments = split_policy(text)
        text_fragments = windows(text_fragments, window_size=window_size)
        fragments.extend(text_fragments)
        groups.extend([i]*len(text_fragments))
        if authors is not None:
            authors_fragments.extend([authors[i]] * len(text_fragments))

    if authors is not None:
        return fragments, authors_fragments, groups

    return fragments, groups


def tokenize(text):
    unmod_tokens = nltk.word_tokenize(text)
    return ([token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)])


# ------------------------------------------------------------------------
# feature extraction methods
# ------------------------------------------------------------------------
def _features_function_words_freq(documents, lang):
    """
    Extract features as the frequency (x1000) of the function words used in the documents
    :param documents: a list where each element is the text (string) of a document
    :return: a np.array of shape (D,F) where D is len(documents) and F is len(function_words)
    """
    features = []
    function_words = get_function_words(lang)

    for text in documents:
        mod_tokens = tokenize(text)
        freqs = nltk.FreqDist(mod_tokens)
        nwords = len(mod_tokens)
        funct_words_freq = [1000. * freqs[function_word] / nwords for function_word in function_words]
        features.append(funct_words_freq)

    return np.array(features)


def _features_conjugations_freq(documents, lang):
    features = []
    conjugations = get_conjugations(lang)

    for text in documents:
        mod_tokens = tokenize(text)
        conjugation_tokens = []
        for conjugation in conjugations:
            conjugation_tokens.extend([conjugation for token in mod_tokens if token.endswith(conjugation) and len(token)>len(conjugation)])
        freqs = nltk.FreqDist(conjugation_tokens)
        nwords = len(mod_tokens)
        conjugation_freq = [1000. * freqs[conjugation] / nwords for conjugation in conjugations]
        features.append(conjugation_freq)

    return np.array(features)


def _features_Mendenhall(documents, upto=23):
    """
    Extract features as the frequency (x1000) of the words' lengths used in the documents,
    following the idea behind Mendenhall's Characteristic Curve of Composition
    :param documents: a list where each element is the text (string) of a document
    :return: a np.array of shape (D,F) where D is len(documents) and F is len(range of lengths considered)
    """
    features = []
    for text in documents:
        unmod_tokens = nltk.word_tokenize(text)
        mod_tokens = ([token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)])
        nwords = len(mod_tokens)
        tokens_len = [len(token) for token in mod_tokens]
        tokens_count = []
        for i in range(1, upto):
            tokens_count.append(1000.*(sum(j>= i for j in tokens_len))/nwords)
        features.append(tokens_count)
    return np.array(features)


def _features_sentenceLengths(documents, downto=3, upto=70):
    """
    Extract features as the length of the sentences, ie. number of words in the sentence.
    :param documents: a list where each element is the text (string) of a document
    :param downto: minimal length considered
    :param upto: maximum length considered
    :return: a np.array of shape (D,F) where D is len(documents) and F is len(range of lengths considered)
    """
    features = []
    for text in documents:
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        nsent = len(sentences)
        sent_len = []
        sent_count = []
        for sentence in sentences:
            unmod_tokens = nltk.tokenize.word_tokenize(sentence)
            mod_tokens = ([token for token in unmod_tokens if any(char.isalpha() for char in token)])
            sent_len.append(len(mod_tokens))
        for i in range(downto, upto):
            sent_count.append(1000.*(sum(j>= i for j in sent_len))/nsent)
        features.append(sent_count)
    return np.array(features)




def _features_tfidf(documents, tfidf_vectorizer=None, min_df = 1, ngrams=(1,1)):
    """
    Extract features as tfidf matrix extracted from the documents
    :param documents: a list where each element is the text (string) of a document
    :return: a tuple M,V, where M is an np.array of shape (D,F), with D being the len(documents) and F the number of
    distinct words; and V is the TfidfVectorizer already fit
    """
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=min_df, ngram_range=ngrams)
        tfidf_vectorizer.fit(documents)

    features = tfidf_vectorizer.transform(documents)

    return features, tfidf_vectorizer


def _features_ngrams(documents, ns=[4, 5], ngrams_vectorizer=None, min_df = 10, preserve_punctuation=True):
    doc_ngrams = ngrams_extractor(documents, ns, preserve_punctuation)
    return _features_tfidf(doc_ngrams, tfidf_vectorizer=ngrams_vectorizer, min_df = min_df)


def ngrams_extractor(documents, ns=[4, 5], preserve_punctuation=True):
    if not isinstance(ns, list): ns=[ns]
    ns = sorted(np.unique(ns).tolist())

    list_ngrams = []
    for doc in documents:
        if preserve_punctuation==False:
            doc = ' '.join(tokenize(doc))
        doc_ngrams = []
        for ni in ns:
            doc_ngrams.extend([doc[i:i + ni].replace(' ','_') for i in range(len(doc) - ni + 1)])

        list_ngrams.append(' '.join(doc_ngrams))

    return list_ngrams


def _feature_selection(X, y, tfidf_feat_selection_ratio):
    nF = X.shape[1]
    num_feats = int(tfidf_feat_selection_ratio * nF)
    feature_selector = SelectKBest(chi2, k=num_feats)
    X = feature_selector.fit_transform(X, y)
    return X, feature_selector

def _tocsr(X):
    return X if issparse(X) else csr_matrix(X)


class FeatureExtractor:

    def __init__(self,
                 function_words_freq=None,
                 conjugations_freq=None,
                 features_Mendenhall=True,
                 features_sentenceLengths=True,
                 wordngrams=False,
                 tfidf_feat_selection_ratio=1.,
                 n_wordngrams=(1, 1),
                 charngrams=False,
                 n_charngrams=[4, 5],
                 preserve_punctuation=True,
                 split_documents=False,
                 split_policy = split_by_endline,
                 normalize_features=True,
                 window_size = 5,
                 verbose=True):
        """
        Loads the documents contained in path applying a number of feature extraction policies. The directory is assumed to
        contain files named according to <author>_<text_name>.txt plus two special files EpistolaXIII_1.txt and
        EpistolaXIII_2.txt concerning the two documents whose authorship attribution is to be determined.
        :param path: the path containing the texts, each named as <author>_<text_name>.txt
        :param function_words_freq: add the frequency of function words as features
        :param features_Mendenhall: add the frequencies of the words' lengths as features
        :param wordngrams: add the tfidf as features
        :param split_documents: whether to split text into smaller documents or not (currenty, the policy is to split by '\n').
        Currently, the fragments resulting from the split are added to the pool of documents (i.e., they do not replace the
        full documents, which are anyway retained).
        :param split_policy: a callable that implements the split to be applied (ignored if split_documents=False)
        :param window_size: the size of the window in case of sliding windows policy
        :param verbose: show information by stdout or not
        :return: np.arrays or csr_matrix (depending on whether tfidf is activated or not) X, y, EP1, EP2, where X is the
        matrix of features for the training set and y are the labels (np.array);
        EP1 and EP2 are the matrix of features for the epistola 1 (first row) and fragments (from row 2nd to last) if
        split_documents=True) and 2 (similar)
        """
        self.function_words_freq = function_words_freq
        self.conjugations_freq = conjugations_freq
        self.features_Mendenhall = features_Mendenhall
        self.features_sentenceLengths = features_sentenceLengths
        self.tfidf = wordngrams
        self.tfidf_feat_selection_ratio = tfidf_feat_selection_ratio
        self.wordngrams = n_wordngrams
        self.ngrams = charngrams
        self.ns = n_charngrams
        self.preserve_punctuation = preserve_punctuation
        self.split_documents = split_documents
        self.split_policy = split_policy
        self.normalize_features=normalize_features
        self.window_size = window_size
        self.verbose = verbose


    def fit_transform(self, positives, negatives):
        documents = positives + negatives
        authors = [1]*len(positives) + [0]*len(negatives)
        n_original_docs = len(documents)
        groups = list(range(n_original_docs))

        if self.split_documents:
            doc_fragments, authors_fragments, groups_fragments = splitter(documents, authors,
                                                        split_policy=self.split_policy,
                                                        window_size=self.window_size)
            documents.extend(doc_fragments)
            authors.extend(authors_fragments)
            groups.extend(groups_fragments)
            self._print('splitting documents: {} documents'.format(len(doc_fragments)))

        # represent the target vector
        y = np.array(authors)
        groups = np.array(groups)

        # initialize the document-by-feature vector
        X = np.empty((len(documents), 0))

        # dense feature extraction functions
        if self.function_words_freq:
            X = self._addfeatures(X, _features_function_words_freq(documents, self.function_words_freq))
            self._print('adding function words features: {} features'.format(X.shape[1]))

        if self.conjugations_freq:
            X = self._addfeatures(X, _features_conjugations_freq(documents, self.conjugations_freq))
            self._print('adding conjugation features: {} features'.format(X.shape[1]))

        if self.features_Mendenhall:
            X = self._addfeatures(X, _features_Mendenhall(documents))
            self._print('adding Mendenhall words features: {} features'.format(X.shape[1]))

        if self.features_sentenceLengths:
                X = self._addfeatures(X, _features_sentenceLengths(documents))
                self._print('adding sentence lengths features: {} features'.format(X.shape[1]))

        # sparse feature extraction functions
        if self.tfidf:
            X_features, vectorizer = _features_tfidf(documents, ngrams=self.wordngrams)
            self.tfidf_vectorizer = vectorizer

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                X_features, feat_sel = _feature_selection(X_features, y, self.tfidf_feat_selection_ratio)
                self.feat_sel_tfidf = feat_sel

            X = self._addfeatures(_tocsr(X), X_features)
            self._print('adding tfidf words features: {} features'.format(X.shape[1]))

        if self.ngrams:
            X_features, vectorizer = _features_ngrams(documents, self.ns,
                                                      preserve_punctuation=self.preserve_punctuation)
            self.ngrams_vectorizer = vectorizer

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                X_features, feat_sel = _feature_selection(X_features, y, self.tfidf_feat_selection_ratio)
                self.feat_sel_ngrams = feat_sel

            X = self._addfeatures(_tocsr(X), X_features)
            self._print('adding ngrams character features: {} features'.format(X.shape[1]))

        # print summary
        if self.verbose:
            print(
                'load_documents: function_words_freq={} features_Mendenhall={} tfidf={}, split_documents={}, split_policy={}'
                .format(self.function_words_freq, self.features_Mendenhall, self.tfidf, self.split_documents,
                        self.split_policy.__name__))
            print('number of training (full) documents: {}'.format(n_original_docs))
            print('X shape (#documents,#features): {}'.format(X.shape))
            print('y prevalence: {}/{} {:.2f}%'.format(y.sum(),len(y),y.mean() * 100))
            print()

        return X, y, groups


    def transform(self, test, return_fragments=False, window_size=-1, avoid_splitting=False):
        test = [test]
        if window_size==-1:
            window_size = self.window_size

        if self.split_documents and not avoid_splitting:
            tests, _ = splitter(test, split_policy=self.split_policy, window_size=window_size)
            test.extend(tests)

        # initialize the document-by-feature vector
        TEST = np.empty((len(test), 0))

        # dense feature extraction functions
        if self.function_words_freq:
            TEST = self._addfeatures(TEST, _features_function_words_freq(test, self.function_words_freq))
            self._print('adding function words features: {} features'.format(TEST.shape[1]))

        if self.conjugations_freq:
            TEST = self._addfeatures(TEST, _features_conjugations_freq(test, self.conjugations_freq))
            self._print('adding conjugation features: {} features'.format(TEST.shape[1]))

        if self.features_Mendenhall:
            TEST = self._addfeatures(TEST, _features_Mendenhall(test))
            self._print('adding Mendenhall words features: {} features'.format(TEST.shape[1]))

        if self.features_sentenceLengths:
            TEST = self._addfeatures(TEST, _features_sentenceLengths(test))
            self._print('adding sentence lengths features: {} features'.format(TEST.shape[1]))

        # sparse feature extraction functions
        if self.tfidf:
            ep1_features, _ = _features_tfidf(test, self.tfidf_vectorizer)

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                ep1_features = self.feat_sel_tfidf.transform(ep1_features)

            TEST = self._addfeatures(_tocsr(TEST), ep1_features)
            self._print('adding tfidf words features: {} features'.format(TEST.shape[1]))

        if self.ngrams:
            ep1_features, _ = _features_ngrams(test, self.ns, ngrams_vectorizer=self.ngrams_vectorizer,
                                               preserve_punctuation=self.preserve_punctuation)

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                ep1_features = self.feat_sel_ngrams.transform(ep1_features)

            TEST = self._addfeatures(_tocsr(TEST), ep1_features)
            self._print('adding ngrams words features: {} features'.format(TEST.shape[1]))

        # print summary
        if self.verbose:
            print(
                'load_documents: function_words_freq={} features_Mendenhall={} tfidf={}, split_documents={}, split_policy={}'
                .format(self.function_words_freq, self.features_Mendenhall, self.tfidf, self.split_documents,
                        self.split_policy.__name__))
            print('test shape:', TEST.shape)
            print()

        if return_fragments:
            return TEST, test[1:]
        else:
            return TEST


    def _addfeatures(self, X, F):
        # plt.matshow(F[:25])
        # plt.show()
        if self.normalize_features:
            normalize(F, axis=1, copy=False)

        if issparse(F):
            return hstack((X, F))  # sparse
        else:
            return np.hstack((X, F))  # dense


    def _print(self, msg):
        if self.verbose:
            print(msg)

