# cs224uSNLI
CS224U Final Project, Spring 2019

## Setup the Environment
1. Follow instructions for setting up the CS224U environment
2. Run `conda install allennlp -c pytorch -c allennlp -c conda-forge`
3. Run `pip install allennlp`

## Setup the Adversarial Datasets
1. Download the ADDAMOD and SUBOBJSWAP data sets from: https://www.dropbox.com/s/lev4327zn833dle/SharingDataResource.zip?dl=0
2. Download the "Breaking SNLI" word-replacement data set from: https://github.com/BIU-NLP/Breaking_NLI
3. Setup this structure in your project directory:
##### /path/to/project_home/data/
* snli_1.0
    * snli_1.0_dev.jsonl
    * etc
    * etc
* snli_adversaries
    * ADDAMOD <-- from Step 1 above
        * add_amod(dev).tsv
        * add_amod(train).tsv
    * SUBOBJSWAP <-- from Step 1 above
        * sub_obj_swap(dev).tsv
        * sub_obj_swap(train).tsv
    * breaking_README.txt
    * SUB_ADD_README.txt
    * dataset.jsonl <-- from Step 2 above

## Pre-Processing
1. For faster training and testing, it is advised that you run the `preprocessing.py`
script on the data. This parses the sentences for their SRL tags in advance.


## Setup the LSTM GloVe + SRL Model

1. Activate the cs224u standard virtual environment
2. Run `jupyter notebook adversarial_snli.ipynb`
3. Modify the filepaths as needed, in particular the GLOVE_HOME, and near the
bottom of the notebook, the various training and evaluation runs which point to
specific data files.
4. Run all cells.

## Baseline Model
1. Modify the code such that `glove.5B.100d.txt` is used for the `glove_lookup`
2. Alter the `glove_srl_phi` function to return GloVe lookup tensors only, not the
concatenated GloVe + SRL tensors. 