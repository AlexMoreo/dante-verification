#!/usr/bin/env bash
set -x

corpus='../MedLatin'

if [ ! -d $corpus ]; then
  curl -0 http://hlt.isti.cnr.it/medlatin/MedLatin.zip -o ../MedLatin.zip
  unzip ../MedLatin.zip -d ../
  rm ../MedLatin.zip
fi

PY="python3 author_identification.py"
MedLatin1="../MedLatin/Corpora/MedLatin1"
MedLatin2="../MedLatin/Corpora/MedLatin2"
EP1="../MedLatin/Epistle/EpistolaXIII_1.txt"
EP2="../MedLatin/Epistle/EpistolaXIII_2.txt"

$PY $MedLatin1 ALL --unknown $EP1 --loo --log ./results_EP1.txt
$PY $MedLatin2 ALL --unknown $EP2 --loo --log ./results_EP2.txt
