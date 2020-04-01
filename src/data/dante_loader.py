import os
from os.path import join
import collections


# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
def remove_pattern(doc, start_symbol, end_symbol, counter):
    assert counter[start_symbol] == counter[end_symbol], 'wrong number of {}{} found'.format(start_symbol,end_symbol)
    search = True
    while search:
        start = doc.find(start_symbol)
        if start > -1:
            end = doc[start + 1:].find(end_symbol)
            doc = doc[:start] + doc[start + 1 + end + 1:]
        else:
            search = False
    return doc


# removes citations in format:
#    *latino*
#    {volgare}
def remove_citations(doc):
    counter = collections.Counter(doc)
    doc = remove_pattern(doc, start_symbol='*', end_symbol='*', counter=counter)
    doc = remove_pattern(doc, start_symbol='{', end_symbol='}', counter=counter)
    return doc


def load_latin_corpus(path, positive_author='Dante', unknown_target=None, train_skip_prefix='Epistola'):
    """
    Function used to load the Corpus I and Corpus II for authorship verification (and validation) of the Epistola XIII.
    The corpus is assumed to contain files named according to <author>_<text_name>.txt.
    :param path: the path containing the texts, each named as <author>_<text_name>.txt
    :param positive_author: the author that defines the positive class for verification
    :param unknown_target: if specified, is the path to the unknown document whose paternity is to be check (w.r.t.
    the positive_author)
    :param train_skip_prefix: specify a prefix for documents that should be skipped
    :return: a tuple containing the positive documents, negative documents, paths to positive documents, paths to
    negative documents, and the unknown document if that was specified (otherwise an empty list)
    """
    # load the training data (all documents but Epistolas 1 and 2)
    positive, negative = [], []
    files_positive, files_negative = [], []

    authors = []
    ndocs=0
    for file in os.listdir(path):
        if file.startswith(train_skip_prefix): continue
        if f'{path}/{file}' == unknown_target: continue
        file_name = file.replace('.txt','')
        author, textname = file_name.split('_')
        text = open(join(path,file), encoding= "utf8").read()
        text = remove_citations(text)

        if author == positive_author:
            positive.append(text)
            files_positive.append(file)
        else:
            negative.append(text)
            files_negative.append(file)
        authors.append(author)
        ndocs += 1

    # load the unknown document (if requested))
    if unknown_target:
        unknown = open(unknown_target, encoding="utf8").read()
        unknown = [remove_citations(unknown)]
    else:
        unknown = []
    return positive, negative, files_positive, files_negative, unknown


def list_authors(path, skip_prefix, skip_authors=['Misc']):
    authors = [file.split('_')[0] for file in os.listdir(path) if not file.startswith(skip_prefix)]
    authors = [author for author in authors if author not in skip_authors]
    return sorted(set(authors))




