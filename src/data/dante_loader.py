import os
from os.path import join
import re
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

def load_texts(path, positive_author='Dante', unknown_target=None):
    # load the training data (all documents but Epistolas 1 and 2)
    positive,negative = [],[]
    authors   = []
    ndocs=0
    for file in os.listdir(path):
        if file.startswith('EpistolaXIII_'): continue
        file_clean = file.replace('.txt','')
        author, textname = file_clean.split('_')[0],file_clean.split('_')[1]
        text = open(join(path,file), encoding= "utf8").read()
        text = remove_citations(text)

        if author == positive_author:
            positive.append(text)
        else:
            negative.append(text)
        authors.append(author)
        ndocs+=1

    # load the test data (Epistolas 1 and 2)
    if unknown_target:
        if isinstance(unknown_target, str):
            unknown_target = [unknown_target]
        unknowns = []
        for unknown_text in unknown_target:
            unknown = open(join(path, unknown_text), encoding="utf8").read()
            unknown = remove_citations(unknown)
            unknowns.append(unknown)
        if len(unknowns) == 1: unknowns = unknowns[0]
        return positive, negative, unknowns

    else:
        return positive, negative


def list_texts(path):
    authors   = {}
    for file in os.listdir(path):
        if file.startswith('EpistolaXIII_'): continue
        file_clean = file.replace('.txt','')
        author, textname = file_clean.split('_')[0],file_clean.split('_')[1]
        if author not in authors:
            authors[author] = []
        authors[author].append(textname)

    author_order = sorted(authors.keys())
    for author in author_order:
        print('{}:\t{}'.format(author,', '.join(authors[author])))







