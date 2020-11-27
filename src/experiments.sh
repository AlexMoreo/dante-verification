#!/usr/bin/env bash
set -x

corpus='../MedLatin'

if [ ! -d $corpus ]; then
  curl -0 http://hlt.isti.cnr.it/medlatin/MedLatin.zip -o ../MedLatin.zip
  unzip ../MedLatin.zip -d ../
  rm ../MedLatin.zip
fi

PYLOO="python3 author_identification_loo.py"
PYUNK="python3 author_identification_unknown.py"

MedLatin1="../MedLatin/Corpora/MedLatin1"
MedLatin2="../MedLatin/Corpora/MedLatin2"

EPXIII1="../MedLatin/Epistle/EpistolaXIII_1.txt"
EPXIII2="../MedLatin/Epistle/EpistolaXIII_2.txt"
EPXIV="../Epistola_ArigoVII.txt"

for learner in lr svm mnb ; do
  $PYLOO $MedLatin1 ALL --learner $learner --log ../results/resultsLOO_EP1_$learner.txt
  $PYLOO $MedLatin2 ALL --learner $learner --log ../results/resultsLOO_EP2_$learner.txt

  $PYUNK $MedLatin1 Dante $EPXIII1 --learner $learner --log ../results/resultsUNK_EP13_1_$learner.txt
  $PYUNK $MedLatin2 Dante $EPXIII2 --learner $learner --log ../results/resultsUNK_EP13_2_$learner.txt
  $PYUNK $MedLatin1 Dante $EPXIV --learner $learner --log ../results/resultsUNK_EP14_$learner.txt
done

