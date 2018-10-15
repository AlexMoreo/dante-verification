import os
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_documents(path, by_sentences=False):
    #read documents
    docs,y = [],[]
    for file in os.listdir(path):
        if file.startswith('EpistolaXIII_'): continue
        file_clean = file.replace('.txt','')
        author, textname = file_clean.split('_')[0],file_clean.split('_')[1]
        if by_sentences:
            lines = open(join(path, file)).readlines()
            docs.extend(lines)
            if author == 'Dante':
                y.extend([1] * len(lines))
            else:
                y.extend([0] * len(lines))
        else:
            docs.append(open(join(path,file)).read())
            if author == 'Dante':
                y.append(1)
            else:
                y.append(0)

    if not by_sentences:
        y = y + y
        docs = docs + docs

    if by_sentences:
        ep1 = open(join(path, 'EpistolaXIII_1.txt')).readlines()
        ep2 = open(join(path, 'EpistolaXIII_2.txt')).readlines()
    else:
        ep1 = [open(join(path, 'EpistolaXIII_1.txt' )).read()]
        ep2 = [open(join(path, 'EpistolaXIII_2.txt')).read()]

    # document representation
    tfidf = TfidfVectorizer(sublinear_tf=True)
    X = tfidf.fit_transform(docs)
    y = np.array(y)
    Epistola1 = tfidf.transform(ep1)
    Epistola2 = tfidf.transform(ep2)

    print('documents read, shape={}'.format(X.shape))
    # print(y)

    return X, y, Epistola1, Epistola2


