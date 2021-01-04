# MEmoLon – The Multilingual Emotion Lexicon

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3779901.svg)](https://doi.org/10.5281/zenodo.3779901)


This is the main repository for our ACL 2020 paper [Learning and Evaluating Emotion Lexicons for 91 Languages](https://arxiv.org/abs/2005.05672).

## Overview
Data and code for this research project are distributed across different places. This github repository serves as landing page linking to other relevant sites. It also contains the code necessary to re-run our experiments and analyses.  Releases of this repository are archived as Zenodo records under [DOI 10.5281/zenodo.3779901](https://doi.org/10.5281/zenodo.3779901). While this repository contains our codebase and experimental results, the generated lexicon is archived in an second Zenodo record under [DOI 10.5281/zenodo.3756606](https://doi.org/10.5281/zenodo.3756606) due to its size.

### Links 
* [Zenodo record of this repository](https://doi.org/10.5281/zenodo.3779901)

* [Zenodo record of the lexicon](https://doi.org/10.5281/zenodo.3756606)

* [arXiv version of the paper](https://arxiv.org/abs/2005.05672) 

* ACL Anthology version of the paper (TBA)

  


## The Lexicon
We created emotion lexicons for 91 languages, each one covers eight emotional variables and  comprises over 100k word entries. There are several versions of the lexicons, the difference being the choice of the expansion model: There is a linear regression baseline and three  versions of neural network models. The *main* version of our lexicons (the version we refer to in the main experiments of our paper and the one we would recommend to use) is referred to as as **MTL_grouped** (applying multi-task learning within two groups of our target variables).   **If you are mainly interested in our lexicons,  download [this](https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1) zip file (2.2GB).**  It contains 91 tsv files which are named `<iso language code>.tsv`. Please refer to the [description of the Zenodo record](https://doi.org/10.5281/zenodo.3756606) for more details.



## The Experimental Results

The analyses and results we present in the paper can be found in `/memolon/analyses` in form of jupyter notebooks and csv / json files. The names of the notebooks follow the section names in the paper. 



## The Codebase

If you are interested in the implementation of our methodology, replicating the lexicon creation or re-running our analyses, this section describes how to work with our code.



### Set-Up 

Make sure you have `conda` installed on your machine.  We ran the code on Debian 9. Necessary steps may differ across operating systems. 

Clone this repository, `cd` into the project's root directory, and run the following commands.

```
conda create --name "memolon2020" python=3.8 pip
conda activate memolon2020
pip install -r requirements.txt
source activate.src
```

The last line configures your `PYTHONTPATH`.



### Re-Running the Lexicon Generation

Recreating the lexicons from scratch requires the Source lexicon, data splits, and the translation tables for all 91 languages. The data split (word lists in `train.txt`, `dev.txt`, and `test.txt`  in `/memolon/data/Source`)   as well as the translation tables (see content of `/memolon/data/TranslationTables`)  are already included in this repository. So, you only have to download the source lexicon. There are two files:

 * Get the file [Ratings_Warriner_et_al.csv](https://github.com/JULIELab/XANEW/blob/master/Ratings_Warriner_et_al.csv)  (commit b1ed97e from 11 Nov 2019) and place it in `/memolon/data/Source`.
 * Get the file [Warriner_BE.tsv](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/Warriner_BE.tsv) (commit dbfa3b9 from 15 Jun 2018) and place it in ``/memolon/data/Source``.

 

The python scripts for creating the lexicons can be found in  `/memolon/src`.  You can either `cd` there and simply run `run_all.sh` or follow the more detailed instructions below. Please take note that the whole process may take several hours. **You do not have to have a GPU to run our code in a reasonable amount of time.**

 * To download the fastText embedding models run `download_embeddings.py` which will download the vec.gz files and place them into `/memolon/data/Embeddings`.
 * To train and use our models to create all four different versions of the target lexicons (`TargetPred`) run the following scripts (or just the one you want to use). They will create the lexicons and place them into the respective subfolder of  `/memolon/data/TargetPred`:
     * `TargetPred_MTLgrouped.py`: Multi-task learning within the two groups (VAD and BE5) but not across both. This is the version we mainly refer to in the paper and **recommend to use**.
     * `TargetPred_MTLall.py`: Multi-task learning among all 8 target variables.
     * `TargetPred_STL.py`: Single-task learning all 8 variables separately.
     * `TargetPred_ridge.py`: The ridge regression baseline.



### Re-Running the Analyses

As stated above, analyses are organized as jupyter notebooks in the folder `/memolon/analyses`. Please note that running some of the notebooks requires data files from other notebooks.  The recommended order of running the notebooks is the following, although other orders are possible as well. 

1. `overview-gold-lexica.ipynb`
2. `silver-evaluation.ipynb`
3. `gold-evaluation.ipynb`
4. `comparison_against_human_reliability.ipynb`
5. `translation_vs_prediction.ipynb`
6. `gold_vs_silver_evaluation.ipynb`
7. `overview-generated-lexicons.ipynb`

Running the silver evaluation is quite simple. You can either generate our lexicons from scratch (see above), or, much easier, download our lexicons from the  [Zenodo record](https://doi.org/10.5281/zenodo.3756606) (see above). Unzip all four versions of the lexicons and place the tsv files in the respective subfolders of `/memolon/data/TargetPred`.

Running the gold evaluation and related analyses requires you to manually collect all the gold datasets listed in the paper. This is a tedious process because they all have different copyright and access restrictions. Please find more detailed instructions below.

 * en1. This is our Source lexicon, see the above section on lexicon generation.
 * en2. Either request the **1999**-version of the Affective Norms for English Words (ANEW) from the [Center for the Study of Emotion and Attention](https://csea.phhp.ufl.edu/Media.html#bottommedia) at the University of Florida, or copy-paste/parse the data from the Techreport  *Bradley, M. M., & Lang, P. J. (1999). Affective Norms for English Words (Anew): Stimuli, Instruction Manual and Affective Ratings (C–1). The Center for Research in Psychophysiology, University of Florida.* Format the data as an tsv file with column headers `word`, `valence`, `arousal`, `dominance` and save it under `/memolon/data/TargetGold/ANEW1999.tsv`.
 * en3. Get the file `Stevenson(2007)-ANEW_emotional_categories.xls` from [Stevenson et al. (2007)](https://doi.org/10.3758/BF03192999) 
 and place it in `/memolon/data/TargetGold`.
 * es1. Get the file `Redondo(2007).xls`  from [Redondo et al. (2007)](https://doi.org/10.3758/BF03193031) and place it `/memolon/data/TargetGold`.
 * es2. Get the file `13428_2015_700_MOESM1_ESM.csv` from [Stadthagen-Gonzalez et al. (2017)](https://doi.org/10.3758/BF03192999) and save it as `/memolon/data/TargetGold/Stadthagen_VA.csv`
 * es3. Get the file `Hinojosa et al_Supplementary materials.xlsx` from [Hinojosa et al. (2015)](https://link.springer.com/article/10.3758%2Fs13428-015-0572-5) and place it in `/memolon/data/TargetGold`.
 * es4. Included in the download for es3.
 * es5. Get the file `13428_2017_962_MOESM1_ESM.csv` from [Stadthagen-Gonzalez et al. (2018)](https://doi.org/10.3758/s13428-017-0962-y) and save it as `/memolon/data/TargetGold/Stadthagen_BE.csv`.
 * es6. Get the file `13428_2016_768_MOESM1_ESM.xls` from [Ferré et al. (2017)](https://doi.org/10.3758/s13428-016-0768-3) ad save it as `/memolon/data/TargetGold/Ferre.xlsx`.
 * de1. Get the file `13428_2013_426_MOESM1_ESM.xlsx` from [Schmidtke et al. (2014)](https://doi.org/10.3758/s13428-013-0426-y) and save it as `/memolon/data/TargetGold/Schmidtke.xlsx`
 * de2. Get the file `BAWL-R.xls` from [Vo et al. (2009)](https://doi.org/10.3758/BRM.41.2.534) which is currently available 
 [here](https://www.ewi-psy.fu-berlin.de/einrichtungen/arbeitsbereiche/allgpsy/Download/BAWL/index.html). You will need to request a password from the authors. Save the file **without password** as `/memolon/data/TargetGold/BAWL-R.xls`. We had to run an automatic file repair when oping it with Excel for the first time.
 * de3. Get the file `LANG_database.txt` from [Kaske and Kotz (2010)](https://doi.org/10.3758/BRM.42.4.987) and place it `/memolon/data/TargetGold`.
 * de4. Get de2 (see above). Then, get the file  `13428_2011_59_MOESM1_ESM.xls` from [Briesemeister et al. (2011)](https://doi.org/10.3758/s13428-011-0059-y) and save it as `/memolon/data/TargetGold/Briesemeister.xls`.
 * pl1. Get the file `data sheet 1.xlsx` from [Imbir (2016)](https://doi.org/10.3389/fpsyg.2016.01081) and save it as  `/memolon/data/TargetGold/Imbir.xlsx`.
 * pl2. Get the file `13428_2014_552_MOESM1_ESM.xlsx` from [Riegel et al. (2015)](https://doi.org/10.3758/s13428-014-0552-1) and save it as `/memolon/data/TargetGold/Riegel.xlsx`
 * pl3. Get pl2 (see above). Then, get the file `S1 Dataset` from [Wierzba et al. (2015)](https://doi.org/10.1371/journal.pone.0132305) 
 and save it as `/memolon/data/TargetGold/Wierzba.xlsx`.
 * zh1. Get CVAW 2.0 from [Yu et al. (2016)](https://doi.org/10.18653/v1/N16-1066) which is distributed via 
 [this website](http://nlp.innobic.yzu.edu.tw/resources/cvaw.html). Use Google Translate to  'translate' the words in `cvaw2.csv` 
 from traditional to simplified Chinese characters (you can batch-translate by copy-pasting multiple words separated by newline directly from the file). Save the modified file as `/memolon/data/TargetGold/cvaw2_simplied.csv`.
 * zh2. Get the file `13428_2016_793_MOESM2_ESM.pdf` from [Yao et al. (2017)](https://doi.org/10.3758/s13428-016-0793-2).  Convert PDF to Excel (there are online tools for that but check the results for correctness) and save as `/memolon/data/TargetGold/Yao.xlsx`.
 * it. Get the data from [Montefinese et al. (2014)](https://doi.org/10.3758/s13428-013-0405-3). The website offers a PDF version
 of the ratings. However, the formatting makes it very difficult to process automatically. Instead, the first author Maria Montefinese provided us with an Excel version.  Save the ratings as `/memolon/data/TargetGold/Montefinese.xls`.
 * pt. Get the file  `13428_2011_131_MOESM1_ESM.xls`  from [Soares et al. (2012)](https://doi.org/10.3758/s13428-011-0131-7). 
 Save it as  `/memolon/data/TargetGold/Soares.xls`.
 * nl. Get the file `13428_2012_243_MOESM1_ESM.xlsx`  from [Moors et al. (2013)](https://doi.org/10.3758/s13428-012-0243-8).
 Save it as  `/memolon/data/TargetGold/Moors.xlsx`.
 * id. Get the file `Data Sheet 1.XLSX` from [Sianipar et al. (2016)](https://doi.org/10.3389/fpsyg.2016.01907). Save it as `/memolon/data/TargetGold/Sianipar.xlsx`
 * el. Get the data from [Palogiannidi et al. (2016)](https://www.aclweb.org/anthology/L16-1458): We downloaded the ratings via the [link](www.telecom.tuc.gr/~epalogiannidi/docs/resources/greek_affective_lexicon.zip)
 provided in the paper on March 13, 2018. The link pointed to zip containing a single file `greek_affective_lexicon.csv`  which we saved under  `/memolon/data/TargetGold`. However, the original link does not work anymore (as of April 22, 2020). We recommend contacting the authors for a replacement. 
 * tr1. Get the file `TurkishEmotionalWordNorms.csv` from [Kapucu et al. (2018)](https://doi.org/10.1177/0033294118814722) 
 which is available [here](https://osf.io/rxtdm/). Place it under `/memolon/data/TargetGold`.
 * tr2. Included in the download for tr1. 
 * hr. Get the file `Supplementary material_Ćoso et al.xlsx` from [Coso et al. (2019)](https://doi.org/10.1177/1747021819834226)
 which is available [here](https://www.ucace.com/links/). Save it as `/memolon/data/TargetGold/Coso.xlsx`.



 ## Citation

If you find this work useful, please cite our paper:

```bib
@misc{buechel2020learning,
    title={Learning and Evaluating Emotion Lexicons for 91 Languages},
    author={Sven Buechel and Susanna Rücker and Udo Hahn},
    year={2020},
    eprint={2005.05672},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    note={To appear in ACL 2020}
}
```





## Contact

Please get in touch via svenericbuechel@gmail.com. 

