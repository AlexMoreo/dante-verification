import os
from os.path import join

# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
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

        if author == positive_author:
            positive.append(text)
        else:
            negative.append(text)
        authors.append(author)
        ndocs+=1

    # load the test data (Epistolas 1 and 2)
    ep1_text = open(join(path, 'EpistolaXIII_1.txt'), encoding="utf8").read()
    ep2_text = open(join(path, 'EpistolaXIII_2.txt'), encoding="utf8").read()

    return positive, negative, ep1_text, ep2_text