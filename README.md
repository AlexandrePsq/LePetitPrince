# LePetitPrince


This repository includes the code of "Le Petit Prince" project.
(LPP = Le Petit Prince)



## Project description

This project is a cross-linguistics study involving NLP and Neurolinguistics experts (NeuroSpin, INRIA, CORNELL, ...).
It aims at better understanding the cortical bases of language comprehension through computational linguistics.
To do so, we acquired the fMRI and MEG activations of subjects listening to the audiobook of "Le Petit Prince".



## Data acquisition

The story is segmented into 9 runs of approximately 10 min each:
- Chapters 1 to 3 --> run 1
- Chapters 4 to 6 --> run 2
- Chapters 7 to 9 --> run 3
- Chapters 10 to 12 --> run 4
- Chapters 13 to 14 --> run 5
- Chapters 15 to 19 --> run 6
- Chapters 20 to 22 --> run 7
- Chapters 23 to 25 --> run 8
- Chapters 26 to 27 --> run 9


This study includes:
- 40 fMRI in English
- 40 fMRI in French
- 20 MEG in French
each of 90 min long.
Native English speaker were scanned at Cornell University on a Discovery MR750, GE Healthcare, 3T MRI scanner (for details, see Bhattasali et al., 2019). The functional images (repetition time=2s; resolution=3.5x3.5x3.5mm) were spatially normalised to the MNI–152 template using the AFNI packages, and processed using multi-echo independent components analysis (Kundu et al., 2012) .
Naive French speaker were scanned at NeuroSpin on ????.


This huge dataset will be shared through neurovault.


## Methodology

We abode by the following methodology:

**For fMRI**:

Selection and implementation of different Language Models (GloVe, LSTM, BERT/CamemBERT, GPT-2).

Analysis pipeline:

Generation of deep-representations from the text (or audio) of "Le Petit Prince" thanks to the selected models.
Construction of a design-matrix by concatenation of the representations associated with the different models of interest.
Nested cross-validated Ridge-regression between our design-matrix and the fMRI data (transformed thanks to Nilearn).
The nested cross-validation (NCV) is composed of a main cross-validation (CV) over the R2 values, and for each split of the main 
cross-validation, another cross-validation is done to determined the regularization (hyperparameter) of the ridge-regression.
If needed (and before the regression), a compression of the representations is done during the CV.
We then concatenate the dataframe of compressed representations with an onset file and convolve the newly constructed dataframe 
with an 'hrf' kernel to get the regressors that will be aligned to the fMRI data thanks to the regression.

**For MEG**:

(Not done yet)



## Data architecture

Due to the high amount of data and analysis, this project data-code-derivatives recquire to be organized in a intuitively way.

Here you have a glance at the overall architecture:

<pre>
├── <b>paradigm</b> <i>(experiences information, stimuli)</i>
├── <b>oldstuff</b> <i>(oldscripts, personnal data/code, ...)</i>
├── <b>code</b> <i>(all the code of all the analysis)</i>
│   ├── <b>MEG</b> <i>(code of the MEG analysis pipeline)</i>
│   └── <b>fMRI</b> <i>(code of the fMRI analysis pipeline)</i>
│
│         └── <b>utilities</b> <i>(utilities functions: parameters settings, splitter for CV, ...)</i>
├── <b>data</b> <i>(all the raw data acquired from sources)</i>
│   ├── <b>fMRI</b> <i>(fMRI data, 9 runs per subject)</i>
│   │   └── <b>english</b>
│   │       └── <b>sub-057</b>
│   │           └── <b>func</b>
│   ├── <b>wave</b> <i>(wave files data, 9 runs, data for models training)</i>
│   │   ├── <b>english</b>
│   │   └── <b>french</b>
│   └── <b>text</b> <i>(text data, raw text, division in 9 runs, onsets/offsets for each runs, data for models training)</i>
│          ├── <b>english</b>
│          │   ├── <b>lstm_training</b>
│          │   └── <b>onsets-offsets</b>
│          └── <b>french</b>
└── <b>derivatives</b> <i>(results of the code above)</i>
      ├── <b>MEG</b>
      └── <b>fMRI</b> <i>(results from the fMRI pipeline in code/fMRI/)</i>
              ├── <b>deep-representations</b> <i>(deep representation dataframes extracted from the models activity)</i>
              │  ├── <b>english</b>
              │  │  └──  <b> model_of_interest </b>
              │  │         ├── <b> deep_representation_run_1.csv </b>
              │  │         ├──  ...
              │  │         └── <b> deep_representation_run_<i>n</i>.csv </b>
              │  └── <b>french</b>
              └── <b>maps</b> <i>Maps deriving from our pipeline</i>
                    ├── <b>english</b>
                    └── <b>french</b>
</pre>

To give more insights on the three main parts of the project:

- **code**
    - MEG data analysis pipeline
    - fMRI data analysis pipeline
    #### TO BE MODIFIED
        - raw-features.py *(generate raw-features and concatenate them with onsets for a given model)*
        - features.py *(convolve the raw-features with a 'hrf' kernel for a given model)*
        - design-matrices.py *(concatenate features of different models of interest)*
        - glm-indiv.py *(GLM model fitted on fMRI data with a design-matrix)*
        - ridge-indiv.py *(Ridge model fitted on fMRI data with a design-matrix)*
        - dodo.py *(this files in the python analog of makefile script)*
    - utilities
        - utils.py *(utility functions)*
        - settings.py *(where we change the parameters)*
        - first_level_analysis.py *(functions for first level analysis of fMRI)*
        - splitter.py *(splitter for multi-groups Ridge CV)*

- **data**
    - we have the fMRI data organized following the BIDS standard except for the name of the final file
    - the MEG should be added in a few months
    - there is the text of LPP, the text is divided into 9 runs, the original onset-offsets of LPP and training data for models
    - wave files, meaning the content of the audio book with the textgrid files and training data for models

- **derivatives**
    - MEG
    - fMRI (to prevent useless use of memory, we only save deep-model representations as well as the final maps output)



## Executing scripts

First, you will need to upload your deep-representation matrix as a .csv file or numpy array in <pre>derivatives/<i>source</i>/deep-representations/<i>language</i>/<i>model_name</i>/</pre>.


### fMRI pipeline

To run the fMRI pipeline, first fill a yaml template specifying the parameters of your analysis and call `main.py`.


### Analysis


#### TO BE MODIFIED
Available analysis so far:
- scatter plot comparison of r2 distributions per ROI in the brain for 2 given models

To run such an analysis, you should first fill in the `analysis.yaml` file with the name of the model you want to study and the name of the study that this scatter plot is suppose to enlighten (e.g. syntax VS semantic).
Then run the following command line:

<pre>
cd $LPP
cd code/fMRI
python analysis.py $LPP/code/analysis.yaml
</pre>
