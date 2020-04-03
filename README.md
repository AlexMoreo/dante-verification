# Authorship Verification for Medieval Latin 

Code to reproduce the experiments reported in the papers
["The Epistle to Cangrande Through the Lens of Computational Authorship Verification"](https://link.springer.com/chapter/10.1007/978-3-030-30754-7_15)
and 
["L’Epistola a Cangrande al vaglio della Computational Authorship Verification: Risultati preliminari (con una postilla sulla cosiddetta XIV Epistola di Dante Alighieri)"](https://www.academia.edu/42297516/L_Epistola_a_Cangrande_al_vaglio_della_Computational_Authorship_Verification_risultati_preliminari_con_una_postilla_sulla_cosiddetta_XIV_Epistola_di_Dante_Alighieri_in_Nuove_inchieste_sull_Epistola_a_Cangrande_a_c._di_A._Casadei_Pisa_Pisa_University_Press_pp._153-192)

## Requirements:
The experiments have been run using the following packages (older versions might work as well):
* joblib==0.11
* nltk==3.4.5
* numpy==1.18.2
* scikit-learn==0.22.2.post1
* scipy==1.4.1


## Disclaimer:
The dataset is not distributed in this version. We have asked the Editors of each document for permission to publish the corpus.
We are waiting for some of these responses to arrive. 

## Running the Experiments
The script in __./src/author_identification.py__ executes the experiments. This is the script syntax (--help):

```
usage: author_identification.py [-h] [--loo] [--unknown PATH] [--log PATH]
                                CORPUSPATH AUTHOR

Authorship verification for Epistola XIII

positional arguments:
  CORPUSPATH      Path to the directory containing the corpus (documents must
                  be named <author>_<texname>.txt)
  AUTHOR          Positive author for the hypothesis (default "Dante"); set to
                  "ALL" to check every author

optional arguments:
  -h, --help      show this help message and exit
  --loo           submit each binary classifier to leave-one-out validation
  --unknown PATH  path to the file of unknown paternity (default None)
  --log PATH      path to the log file where to write the results (default
                  ./results.txt)
```

The following command line:
```
cd src
python author_identification.py ../Corpora/CorpusI Dante --unknown ../Epistle/EpistolaXIII_1.txt
```

Will use all texts in ../Corpora/CorpusI as training documents to train a verificator for the 
file ../Epistle/EpistolaXIII_1.txt assuming Dante is the positive class (i.e., it will check if, on the
basis of the evidence shown in other Dante's texts, the unknown one belongs to Dante or not). 
The output is probabilistic, informing of the uncertainty that the classifier has in attributing the document
to the positive class. 

Similarly, the command line:
```
cd src
python author_identification.py ../Corpora/CorpusI ALL --loo 
```
will perform a cross-validation of the binary classifier for all authors using all training documents in a leave-one-out (LOO) fashion.

The script will report the results both in the standard output (more elaborated) and in a log file. For example, the last command will produce a log file containing:
```
F1 for ClaraAssisiensis = 0.400
F1 for Dante = 0.957
F1 for GiovanniBoccaccio = 1.000
F1 for GuidoFaba = 0.974
F1 for PierDellaVigna = 0.993
LOO Macro-F1 = 0.865
LOO Micro-F1 = 0.981
```
(Note that small numerical variations with respect to the original papers might occur due to different software versions and as a result from any stochastic underlying process. Those changes should anyway not alter the conclusions derived from the published results.)
