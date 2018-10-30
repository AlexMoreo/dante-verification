import nltk
import numpy as np
import os
from os.path import join

function_words = ["et", "in", "de", "ad", "ut", "cum", "non", "per", "a", "que", "ex", "sed"]

def load_documents(path):
    X, y = [], []
    i=0;
    for file in os.listdir(path):
        if file.startswith('EpistolaXIII_'): continue
        file_clean = file.replace('.txt','')
        author, textname = file_clean.split('_')[0],file_clean.split('_')[1]
        tokens = nltk.word_tokenize(open(join(path,file), encoding= "utf8").read())
        author_tokens = ([token.lower() for token in tokens
                                  if any(char.isalpha() for char in token)])
        freqs= nltk.FreqDist(author_tokens)
        X.append([])
        #print(f"From {textname} by {author}:")
        for function_word in function_words:
            feature= (freqs[function_word]*1000)/len(author_tokens)
            #print(function_word, " = ", freqs[function_word], ", ", feature)
            X[i].append(feature)
        i+=1
        if author == "Dante":
            y.append(1)
        else:
            y.append(0)
            
    
    y= y + y
    X= X + X
    y= np.array(y)
    
    ep = []
    tokens = nltk.word_tokenize(open(join(path, 'EpistolaXIII_2.txt'), encoding= "utf8").read())
    ep2_tokens = ([token.lower() for token in tokens
                                  if any(char.isalpha() for char in token)])
    freqs= nltk.FreqDist(ep2_tokens)
    #print("From Epistola XIII_2:")
    for function_word in function_words:
        feature= (freqs[function_word]*1000/len(ep2_tokens))
        ep.append(feature)
        #print(function_word, " = ", freqs[function_word], ", ", feature)
        ep2 = np.array(ep)

    return X, y, ep2
            
        
