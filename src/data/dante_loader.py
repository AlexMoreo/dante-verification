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


def load_texts(path, positive_author='Dante'):
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
    ep1_text = open(join(path, 'EpistolaXIII_1.txt'), encoding="utf8").read()
    ep2_text = open(join(path, 'EpistolaXIII_2.txt'), encoding="utf8").read()
    ep1_text = remove_citations(ep1_text)
    ep2_text = remove_citations(ep2_text)

    return positive, negative, ep1_text, ep2_text

