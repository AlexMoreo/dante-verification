#!/usr/bin/env bash
set -x

corpus='../MedLatin'
urldata='https://zenodo.org/record/4298503/files/MedLatin.zip'

if [ ! -d $corpus ]; then
  curl -0 $urldata -o ../MedLatin.zip
  unzip ../MedLatin.zip -d ../
  rm ../MedLatin.zip
fi

PYLOO="python3 author_identification_loo.py"
PYUNK="python3 author_identification_unknown.py"

MedLatin1="../MedLatin/Corpora/MedLatinEpi"
MedLatin2="../MedLatin/Corpora/MedLatinLit"

EPXIII1="../MedLatin/Epistle/EpistolaXIII_1.txt"
EPXIII2="../MedLatin/Epistle/EpistolaXIII_2.txt"
EPXIV="../MedLatin/Epistle/Epistola_ArigoVII.txt"

mkdir -p ../results

for learner in lr svm mnb ; do
  $PYLOO $MedLatin1 ALL --learner $learner --log ../results/resultsLOO_EP1_$learner.txt
  $PYLOO $MedLatin2 ALL --learner $learner --log ../results/resultsLOO_EP2_$learner.txt
done

$PYUNK $MedLatin1 Dante $EPXIII1 --learner lr --log ../results/resultsUNK_EP13_1_lr.txt
$PYUNK $MedLatin2 Dante $EPXIII2 --learner lr --log ../results/resultsUNK_EP13_2_lr.txt
$PYUNK $MedLatin1 Dante $EPXIV --learner lr --log ../results/resultsUNK_EP14_lr.txt

