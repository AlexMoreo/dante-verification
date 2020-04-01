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
                      'abuntur', 'ebuntur', 'ientur', 'i', 'isti', 'it', 'istis', 'erunt', 'em', 'eam', 'eas',
                      'ias', 'eat', 'iat', 'eamus', 'iamus', 'eatis', 'iatis', 'eant', 'iant', 'er', 'ear', 'earis',
                      'iaris', 'eatur', 'iatur', 'eamur', 'iamur', 'eamini', 'iamini', 'eantur', 'iantur', 'rem', 'res',
                      'ret', 'remus', 'retis', 'rent', 'rer', 'reris', 'retur', 'remur', 'remini', 'rentur', 'erim',
                      'issem', 'isses', 'isset', 'issemus', 'issetis', 'issent', 'a', 'ate', 'e', 'ete', 'ite', 'are',
                      'ere', 'ire', 'ato', 'eto', 'ito', 'atote', 'etote', 'itote', 'anto', 'ento', 'unto', 'iunto',
                      'ator', 'etor', 'itor', 'aminor', 'eminor', 'iminor', 'antor', 'entor', 'untor', 'iuntor', 'ari',
                      'eri', 'iri', 'andi', 'ando', 'andum', 'andus', 'ande', 'ans', 'antis', 'anti', 'antem', 'antes',
                      'antium', 'antibus', 'antia', 'esse', 'sum', 'est', 'sumus', 'estis', 'sunt', 'eram', 'eras',
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

    f_names = [f'funcw::{f}' for f in function_words]

    return np.array(features), f_names


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

    f_names = [f'conj::{f}' for f in conjugations]

    return np.array(features), f_names


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

    f_names = [f'mendenhall::{c}' for c in range(1,upto)]

    return np.array(features), f_names


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

    f_names = [f'sentlength::{c}' for c in range(downto, upto)]

    return np.array(features), f_names


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


# We have implemented ngrams extration generically, following Sapkota et al. (ref [39] in the PAN 2015 overview), i.e.,
# containing punctuation marks. However, this does not apply to this study since punctuation marks are filtered-out in
# editions of Latin texts.
# More recently, it was shown that character n-grams corresponding to word affixes and including punctuation
# marks are the most significant features in cross-topic authorship attribution [57].
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
        self.feature_names = None


    def fit_transform(self, positives, negatives):
        documents = positives + negatives
        authors = [1]*len(positives) + [0]*len(negatives)
        n_original_docs = len(documents)
        groups = list(range(n_original_docs))
        self.feature_names = []

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
            F, f_names = _features_function_words_freq(documents, self.function_words_freq)
            X = self._addfeatures(X, F)
            self.feature_names.extend(f_names)
            self._print('adding function words features: {} features'.format(X.shape[1]))
        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'

        if self.conjugations_freq:
            F, f_names = _features_conjugations_freq(documents, self.conjugations_freq)
            X = self._addfeatures(X, F)
            self.feature_names.extend(f_names)
            self._print('adding conjugation features: {} features'.format(X.shape[1]))
        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'

        if self.features_Mendenhall:
            F, f_names = _features_Mendenhall(documents)
            X = self._addfeatures(X, F)
            self.feature_names.extend(f_names)
            self._print('adding Mendenhall words features: {} features'.format(X.shape[1]))
        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'

        if self.features_sentenceLengths:
            F, f_names = _features_sentenceLengths(documents)
            X = self._addfeatures(X, F)
            self.feature_names.extend(f_names)
            self._print('adding sentence lengths features: {} features'.format(X.shape[1]))
        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'

        # sparse feature extraction functions
        if self.tfidf:
            X_features, vectorizer = _features_tfidf(documents, ngrams=self.wordngrams)
            self.tfidf_vectorizer = vectorizer
            index2word = {i: w for w, i in vectorizer.vocabulary_.items()}
            f_names = [f'tfidf::{index2word[i]}' for i in range(len(index2word))]

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                X_features, feat_sel = _feature_selection(X_features, y, self.tfidf_feat_selection_ratio)
                self.feat_sel_tfidf = feat_sel
                f_names = [f_names[i] for i in feat_sel.get_support(indices=True)]

            X = self._addfeatures(_tocsr(X), X_features)
            self.feature_names.extend(f_names)
            self._print('adding tfidf words features: {} features'.format(X.shape[1]))

        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'
        if self.ngrams:
            X_features, vectorizer = _features_ngrams(documents, self.ns,
                                                      preserve_punctuation=self.preserve_punctuation)
            self.ngrams_vectorizer = vectorizer
            index2word = {i: w for w, i in vectorizer.vocabulary_.items()}
            f_names = [f'ngram::{index2word[i]}' for i in range(len(index2word))]

            if self.tfidf_feat_selection_ratio < 1.:
                if self.verbose: print('feature selection')
                X_features, feat_sel = _feature_selection(X_features, y, self.tfidf_feat_selection_ratio)
                self.feat_sel_ngrams = feat_sel
                f_names = [f_names[i] for i in feat_sel.get_support(indices=True)]

            X = self._addfeatures(_tocsr(X), X_features)
            self.feature_names.extend(f_names)
            self._print('adding ngrams character features: {} features'.format(X.shape[1]))

        self.feature_names = np.asarray(self.feature_names)

        assert X.shape[1] == len(self.feature_names), f'wrong number of feature names, expected {X.shape[1]} found {len(self.feature_names)}'
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
            F,_=_features_function_words_freq(test, self.function_words_freq)
            TEST = self._addfeatures(TEST, F)
            self._print('adding function words features: {} features'.format(TEST.shape[1]))

        if self.conjugations_freq:
            F,_=_features_conjugations_freq(test, self.conjugations_freq)
            TEST = self._addfeatures(TEST, F)
            self._print('adding conjugation features: {} features'.format(TEST.shape[1]))

        if self.features_Mendenhall:
            F,_ = _features_Mendenhall(test)
            TEST = self._addfeatures(TEST, F)
            self._print('adding Mendenhall words features: {} features'.format(TEST.shape[1]))

        if self.features_sentenceLengths:
            F, _ = _features_sentenceLengths(test)
            TEST = self._addfeatures(TEST, F)
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
        if self.normalize_features:
            normalize(F, axis=1, copy=False)

        if issparse(F):
            return hstack((X, F))  # sparse
        else:
            return np.hstack((X, F))  # dense


    def _print(self, msg):
        if self.verbose:
            print(msg)




if __name__=='__main__':
    from collections import Counter

    # text = 'Magnifico atque uictorioso domino, domino Cani Grandi de la Scala, sacratissimi cesarei principatus in urbe Uerona et ciuitate Uicentie uicario generali, deuotissimus suus Dantes Alagherii, Florentinus natione non moribus, uitam orat per tempora diuturna felicem, et gloriosi nominis perpetuum incrementum.'
    text = 'Magnifico atque uictorioso domino, domino Cani Grandi de la Scala, sacratissimi cesarei principatus in urbe Uerona et ciuitate Uicentie uicario generali, deuotissimus suus Dantes Alagherii, Florentinus natione non moribus, uitam orat per tempora diuturna felicem, et gloriosi nominis perpetuum incrementum. Inclita uestre magnificentie laus, quam fama uigil uolitando disseminat, sic distrahit in diuersa diuersos, ut hos in spem sue prosperitatis attollat, hos exterminii deiciat in terrorem. Huius quidem preconium, facta modernorum exsuperans, tanquam ueri existentia latius, arbitrabar aliquando superfluum. Uerum, ne diuturna me nimis incertitudo suspenderet, uelut Austri regina Ierusalem petiit, uelut Pallas petiit Elicona, Ueronam petii fidis oculis discussurus audita, ibique magnalia uestra uidi, uidi beneficia simul et tetigi; et quemadmodum prius dictorum ex parte suspicabar excessum, sic posterius ipsa facta excessiua cognoui. Quo factum est ut ex auditu solo cum quadam animi subiectione beniuolus prius exstiterim; sed ex uisu postmodum deuotissimus et amicus. Nec reor amici nomen assumens, ut nonnulli forsitan obiectarent, reatum presumptionis incurrere, cum non minus dispares connectantur quam pares amicitie sacramento. Nam si delectabiles et utiles amicitias inspicere libeat, illis persepius inspicienti patebit, preheminentes inferioribus coniugari personas. Et si ad ueram ac per se amicitiam torqueatur intuitus, nonne illustrium summorumque principum plerunque uiros fortuna obscuros, honestate preclaros, amicos fuisse constabit? Quidni, cum etiam Dei et hominis amicitia nequaquam impediatur excessu? Quod si cuiquam, quod asseritur, nunc uideretur indignum, Spiritum Sanctum audiat, amicitie sue participes quosdam homines profitentem. Nam in Sapientia de sapientia legitur, quoniam *infinitus thesaurus est hominibus, quo qui usi sunt, participes facti sunt amicitie Dei*. Sed habet imperitia uulgi sine discretione iudicium; et quemadmodum solem pedalis magnitudinis arbitratur, sic et circa mores uana credulitate decipitur. Nos autem, quibus optimum quod est in nobis noscere datum est, gregum uestigia sectari non decet, quin ymo suis erroribus obuiare tenemur. Nam intellectu ac ratione degentes, diuina quadam libertate dotati, nullis consuetudinibus astringuntur; nec mirum, cum non ipsi legibus, sed ipsis leges potius dirigantur. Liquet igitur, quod superius dixi, me scilicet esse deuotissimum et amicum, nullatenus esse presumptum. Preferens ergo amicitiam uestram quasi thesaurum carissimum, prouidentia diligenti et accurata solicitudine illam seruare desidero. Itaque, cum in dogmatibus moralis negotii amicitiam adequari et saluari analogo doceatur, ad retribuendum pro collatis beneficiis plus quam semel analogiam sequi michi uotiuum est; et propter hoc munuscula mea sepe multum conspexi et ab inuicem segregaui, nec non segregata percensui, dignius gratiusque uobis inquirens. Neque ipsi preheminentie uestre congruum magis comperi magis quam Comedie sublimem canticam, que decoratur titulo Paradisi; et illam sub presenti epistola, tanquam sub epigrammate proprio dedicatam, uobis ascribo, uobis offero, uobis denique recommendo. Illud quoque preterire silentio simpliciter inardescens non sinit affectus, quod in hac donatione plus dono quam domino et honoris et fame conferri potest uideri.Quidni cum eius titulum iam presagiam de gloria uestri nominis ampliandum? Satis actenus uidebar expressisse quod de proposito fuit; sed zelus gratie uestre, quam sitio quasi uitam paruipendens, a primordio metam prefixam urget ulterius. Itaque, formula consumata epistole, ad introductionem oblati operis aliquid sub lectoris officio compendiose aggrediar.'
    print(text)

    # char n-grams
    w=3
    ngrams = [text[i:i+w].replace(' ', '_') for i in range(len(text)-w + 1)]
    print('ngrams')
    print(', '.join(ngrams))
    print(Counter(ngrams).most_common())

    # word n-grams
    w = 2
    words = text.split()
    wngrams = ['_'.join(words[i:i + w]).replace(',','') for i in range(len(words) - w + 1)]
    print('\nwngrams')
    print(', '.join(wngrams))
    print(Counter(wngrams).most_common())

    fn_words = [w if w not in latin_function_words else f"{w}(*)" for w in words]
    print('\nfunction words')
    print(' '.join(fn_words))

    verbal_words = []
    for w in words:
        lcs = sorted(latin_conjugations, key=lambda x: -len(x))
        toadd = w
        for lc in lcs:
            if len(w) <= len(lc): continue
            if w.endswith(lc):
                toadd = w[:-len(lc)] + f'[{lc}]'
                break
        verbal_words.append(toadd)
    print('\nverbal endings')
    print(' '.join(verbal_words))

    print('\nword lengths')
    counter = Counter([len(w.replace(',','')) for w in words])
    total = len(words)
    x,y=[],[]
    cum_req = 0
    print(f'words length\tcount\tfrequency\tcumulative')
    for i in range(1,24):
        x.append(i)
        c = counter[i]
        freq = c / total
        cum_req += freq
        y.append(cum_req)
        if c > 0:
            print(f'{i}\t{c}\t{freq:.2f}\t{cum_req:.2f}')

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.plot(x, y, 'o-')
    # plt.xlabel('word length')
    # plt.ylabel('cumulative frequency')
    # plt.title('')
    # plt.grid()
    # plt.show()


    print('\nsentence length')
    sentences = split_by_sentences(text)
    counter = Counter([len(s.split()) for s in sentences])
    total = len(sentences)
    cum_req = 0
    print(f'words length\tcount\tfrequency\tcumulative')
    dots=True
    rows=0
    for i in range(1,70):
        x.append(i)
        c = counter[i]
        freq = c / total
        cum_req += freq
        if c > 0:
            print(f'{i}\t{c}\t{freq:.3f}\t{cum_req:.2f}')
            dots=True
            rows+=1
        else:
            if dots:
                print(f'...\t...\t...\t...')
                dots=False
    print(counter)
    print('rows',rows)
