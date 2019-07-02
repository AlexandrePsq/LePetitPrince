# LePetitPrince

This repository includes the code of "Le Petit Prince" project.
(LPP = Le Petit Prince)


## Project description

This project is a cross-linguistics study involving NLP and Neurolinguistics experts (NeuroSpin, FAIR, INRIA, CORNELL, ...).
It aims at better understanding the cortical bases of language comprehension through computational linguistics.


## Data acquisition

This study includes:
- 40 fMRI in English
- 40 fMRI in French
- 20 MEG in French
each of 90 min long.

These data were acquired by passive listening of the audiobook of "Le Petit Prince", divided in 9 runs of approximately 10 min each.

This huge dataset will be shared through neurovault.


## Methodology

In order to do so we followed the following methodology:

**For fMRI**:

Selection and implementation of different Language Models.

Analysis pipeline:

Generation of raw-features from the text (or audio) of "Le Petit Prince" thanks to the selected models.
Concatenation of the raw-feature dataframe with an onset file (the result is called raw-features).
Convolution of the newly constructed dataframe with an 'hrf' kernel (the result is called features).
Construction of a design-matrix by concatenation of the features associated with the different models of interest (the result is called design-matrix).
Ridge (cross validated) regression between our design-matrix and the fMRI data (transformed thanks to Nilearn)(the result is called ridge-indiv).

**For MEG**:

(Not done yet)

## Data architecture

Due to the high amount of data and analysis, this project data-code-derivatives recquire to be organized in a intuitively way.
To do so, we first created the script create_architecture.py that will do so automatically (we will see how to execute the script later).

Here you have a glance at the overall architecture:

<pre>
├── <b>paradigm</b> (experiences information, stimuli)
├── <b>oldstuff</b> (oldscripts, personnal data/code, ...)
├── <b>code</b> (all the code of all the analysis)
│   ├── <b>MEG</b> (code of the MEG analysis pipeline)
│   ├── <b>fMRI</b> (code of the fMRI analysis pipeline)
│   ├── <b>models</b> (code related to models initialization/training/generation)
│   │   ├── <b>english</b>
│   │   │   ├── <b>LSTM</b> (LSTM framework)
│   │   │   ├── <b>RMS</b> (Framework for wave properties analysis)
│   │   │   ├── <b>WORDRATE</b> (Framework for simple linguistics properties analysis)
│   │   │   ├── lstm_wikikristina_embedding-size_200_nhid_300_nlayers_1_dropout_01.py (instantiation of a LSTM model)
│   │   │   ├── lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01.py (instantiation of a LSTM model)
│   │   │   └── ...
│   │   └── <b>french</b>
│   └── <b>utilities</b> (utilities functions: parameters settings, splitter for CV, ...)
├── <b>data</b> (all the raw data acquired from sources)
│   ├── <b>fMRI</b> (fMRI data, 9 runs per subject)
│   │   └── <b>english</b>
│   │       └── <b>sub-057</b>
│   │           └── <b>func</b>
│   ├── <b>wave</b> (wave files data, 9 runs, data for models training)
│   │   ├── <b>english</b>
│   │   └── <b>french</b>
│   └── <b>text</b> (text data, raw text, division in 9 runs, onsets/offsets for each runs, data for models training)
│       ├── <b>english</b>
│       │   ├── <b>lstm_training</b>
│       │   └── <b>onsets-offsets</b>
│       └── <b>french</b>
└── <b>derivatives</b> (results of the code above)
    ├── <b>MEG</b>
    └── <b>fMRI</b> (results from the fMRI pipeline in code/fMRI/)
        ├── <b>design-matrices</b> (concatenation of features associated with different models of interest)
        │   └── <b>english</b>
        ├── <b>features</b> (Raw-features convolved with an 'hrf' kernel)
        │   └── <b>english</b>
        ├── <b>glm-indiv</b> (GLM model fitted on fMRI data with a design-matrix)
        │   └── <b>english</b>
        ├── <b>models</b> (trained models)
        │   └── <b>english</b>
        ├── <b>raw_features</b> (Result of a model generation from the text/wave file of LPP, concatenated with the adequate onsets file)
        │   └── <b>english</b>
        └── <b>ridge-indiv</b> (Ridge model fitted on fMRI data with a design-matrix)
            └── <b>english</b>
</pre>

To give more insights on the three main parts of the project:

- **code**
    - MEG data analysis pipeline
    - fMRI data analysis pipeline
        - raw-features.py (generate raw-features and concatenate them with onsets for a given model)
        - features.py (convolve the raw-features with a 'hrf' kernel for a given model)
        - design-matrices.py (concatenate features of different models of interest)
        - glm-indiv.py (GLM model fitted on fMRI data with a design-matrix)
        - ridge-indiv.py (Ridge model fitted on fMRI data with a design-matrix)
        - dodo.py (this files in the python analog of makefile script)
    - utilities
        - utils.py (utility functions)
        - settings.py (where we change the parameters)
        - first_level_analysis.py (functions for first level analysis of fMRI)
        - splitter.py (splitter for multi-groups Ridge CV)
    - models
        - \*XXXX\* : framework associated with a kind of model (e.g. LSTM)
            - model.py (class definition)
            - train.py (train the model)
            - tokenizer.py (adequate tokenizer for the model training and generation)
            - utils.py (utilities functions)
        - \*xxxx\* : instantiation of a given class model

- **data**
    - we have the fMRI data organized following the BIDS standard except for the name of the final file
    - the MEG should be added in a few months
    - there is the text of LPP, the text divided into 9 runs, the original onset-offsets of LPP and training data for models
    - wave files, meaning the content of the audio book with the textgrid files and training data for models

- **derivatives**
    - MEG
    - fMRI (every script of `code/fMRI/` fills a folder of the same name here, the same goes for `code/models/`)


## Executing scripts

### Model training

To train a given model *model_name.py* in a given language *language*, just write:
($LPP is the path to the LPP project home directory)

```
cd $LPP
cd code
python models/*language*/*model_name.py*

```

### fMRI pipeline

To start the fMRI pipeline analysis, first:
    - start by modifying the `code/utilities/settings.py` file with the parameters, paths, subjects and models that you want to study.

```
cd $LPP
cd code/fMRI
doit

```

Normally, the dodo.py will not run a file that has already been run except if it has been modified.
If you still want to do so, run:

```
doit clean
doit forget

```

Running `doit` will first create the adequate architecture and then start the fMRI pipeline.