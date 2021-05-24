# Authorship Verification for Medieval Latin 

This repo was originary meant to reproduce the the experiments reported in the paper
["L’Epistola a Cangrande al vaglio della Computational Authorship Verification: Risultati preliminari (con una postilla sulla cosiddetta XIV Epistola di Dante Alighieri)"](https://www.academia.edu/42297516/L_Epistola_a_Cangrande_al_vaglio_della_Computational_Authorship_Verification_risultati_preliminari_con_una_postilla_sulla_cosiddetta_XIV_Epistola_di_Dante_Alighieri_in_Nuove_inchieste_sull_Epistola_a_Cangrande_a_c._di_A._Casadei_Pisa_Pisa_University_Press_pp._153-192)
(the old version is still accessible [here](https://zenodo.org/record/3903236)). 
Since then, we have improved the experimental protocol (also thanks to the advise of some anonymous reviewers) and added some
functionality (new learners). The new protocol and the new experiments are to be discussed in a paper currently
under review. 

## Requirements:
The experiments have been run using the following packages (older versions might work as well):
* joblib==0.14.1
* nltk==3.4.5
* numpy==1.18.2
* scipy==1.4.1
* sklearn==0.22

## Dataset:
The dataset can be downloaded from [zenodo](https://zenodo.org/record/4298503). 
The script downloads and prepares the dataset automatically.

## Running the Experiments
To run all the experiments simply type:

```
git clone https://github.com/AlexMoreo/dante-verification.git
cd dante-verification/src
chmod +x experiments.sh
./experiments.sh
```

This script will download the dataset the first time it is invoked, and run all experiments.

There are two main applications: [author_identification_loo.py](./src/author_identification_loo.py)
and [author_identification_unknown.py](./src/author_identification_unknown.py). The former carries out
a Leave-One-Out (LOO) evaluation of the verifier, while the second trains a verifier that is then asked to 
verify the authorshipe of a text of unknown paternity. 

The script in __./src/author_identification.py__ executes the experiments. This is the script syntax (--help):

```
usage: author_identification_loo.py [-h] [--log PATH]
                                    [--featsel FEAT_SEL_RATIO]
                                    [--class_weight CLASS_WEIGHT]
                                    [--learner LEARNER]
                                    CORPUSPATH AUTHOR

Authorship verification for MedLatin submit each binary classifier to leave-
one-out validation

positional arguments:
  CORPUSPATH            Path to the directory containing the corpus (documents
                        must be named <author>_<texname>.txt)
  AUTHOR                Positive author for the hypothesis (default "Dante");
                        set to "ALL" to check every author

optional arguments:
  -h, --help            show this help message and exit
  --log PATH            path to the log file where to write the results (if
                        not specified, then the name isautomatically generated
                        from the arguments and stored in ../results/)
  --featsel FEAT_SEL_RATIO
                        feature selection ratio for char- and word-ngrams
  --class_weight CLASS_WEIGHT
                        whether or not to reweight classes' importance
  --learner LEARNER     classification learner (lr, svm, mnb)

```

The following command line:
```
cd src
python author_identification.py ../MedLatin/Corpora/MedLatinEpi ALL
```

Will evaluate the verifier according to a LOO evaluation, independently for each author 
in the corpus (only authors with at least 2 documents are considered -- the LOO cannot be
performed with only one document).
For each author, the performance of the verifier is measured in terms of F1-score (F1) and Accuracy (Acc).
The counters of classification (TP, FP, FN, and TN, as for a 4-cell contingency table) are also listed toguether with the
list of missclassified documents.
Finally, the Macro- and micro-averages for F1 and the mean Accuracy is reported.
The following is an example of the output this script generates:

```
ClaraAssisiensis
	F1 = 1.000
	Acc = 1.000
	TP=4 FP=0 FN=0 TN=290
	Errors for ClaraAssisiensis: 
Dante
	F1 = 0.857
	Acc = 0.990
	TP=9 FP=0 FN=3 TN=282
	Errors for Dante: Dante_epistola1.txt, Dante_epistola12.txt, Dante_epistola4.txt
GiovanniBoccaccio
	F1 = 0.980
	Acc = 0.997
	TP=24 FP=1 FN=0 TN=269
	Errors for GiovanniBoccaccio: Dante_epistola4.txt
GuidoFaba
	F1 = 0.946
	Acc = 0.973
	TP=70 FP=0 FN=8 TN=216
	Errors for GuidoFaba: GuidoFaba_epistola38.txt, GuidoFaba_epistola104.txt, GuidoFaba_epistola102.txt, GuidoFaba_epistola2.txt, GuidoFaba_epistola9.txt, GuidoFaba_epistola48.txt, GuidoFaba_epistola31.txt, GuidoFaba_epistola28.txt
PierDellaVigna
	F1 = 0.986
	Acc = 0.986
	TP=146 FP=4 FN=0 TN=144
	Errors for PierDellaVigna: Misc_epistola25.txt, Misc_epistola1.txt, GuidoFaba_epistola48.txt, ClaraAssisiensis_EpistolaAdSanctamAgnetemDePraga2.txt
LOO Macro-F1 = 0.954
LOO Micro-F1 = 0.969
LOO Accuracy = 0.989
```

The other script, [author_identification_unknown.py](./src/author_identification_unknown.py), is devoted to
train a verifier for a particular author, and check whether or not a text has been written by this author.
The syntax is as follows (run the script with --help to get a detailed list of commands).

```
python3 author_identification_unknown.py <CorpusPath> <Author> <Text> --log <ResultPath>

```

For example, the following command will check if Dante wrote the first part of the "Epistle XIII" as according
to a Logistic Regressor trained (with hyperparamenter tunning by 10fold cross-validation) on the MedLatinEpi corpus:

```
python3 author_identification_unknown.py ../MedLatin/Corpora/MedLatinEpi Dante ../MedLatin/Epistle/EpistolaXIII_1.txt --learner lr --log ../results/resultsUNK_EP13_1_lr.txt
```

The following output will be produced:

```
../MedLatin/Epistle/EpistolaXIII_1.txt: Posterior probability for Dante is 0.3675; 
this means the classifier attributes the text to not-Dante
```

## Results:

The results we have obtained are included in the directory [./results](./results).
The results include the LOO experiments for logistic regression (lr), support vector machines (svm)
and multinomial naive Bayes (mnb), both for MedLatinEpi and for MedLatinLit, and the verification
results for the first and second part of the Epistle to Cangrande and the Epistle to Henry VII.

We also include, in .ods format, the matrices of document-by-author asignments, in which
we color in green the correctly attributed examples and in red the incorrectly attributed examples,
as according to our logistic regressor classifier. 