# LePetitPrince

This repository includes the code of "Le Petit Prince" project.
(LPP = Le Petit Prince)

English, French and Chinese participants were scanned using fMRI while listening to the whole audiobook of the Little Prince (~90min), in their native language.

Natural Language Processing models are used to generate predictors (lexical frequency, syntactic complexity, semantic features, etc.) that are correlated with the brain signals.

The audiobook was segmented into nine audiofiles, presented in successive runs:

- Chapters 1 to 3 --> run 1
- Chapters 4 to 6 --> run 2
- Chapters 7 to 9 --> run 3
- Chapters 10 to 12 --> run 4
- Chapters 13 to 14 --> run 5
- Chapters 15 to 19 --> run 6
- Chapters 20 to 22 --> run 7
- Chapters 23 to 25 --> run 8
- Chapters 26 to 27 --> run 9

The wave files (TODO: add pointer to their location) were annotated for words onset-onsets (TODO: add pointer to the TextGrid files)


### fMRI acquisition

TODO: add the description of the sequences in English and in French

- English Multiecho 

- French Multiecho Siemens 3T

- Chinese


## Data analyses


### fMRI ###

Analysis pipeline:

* Generation of raw-features from the text (or audio) of "Le Petit Prince" from the selected models.
* Concatenation of the raw-feature dataframe with an onset file (the result is called raw-features).
* Convolution of the newly constructed dataframe with an 'hrf' kernel (the result is called features).
* Construction of a design-matrix by concatenation of the features associated with the different models of interest (the result is called design-matrix).
* Ridge (cross validated) regression between our design-matrix and the fMRI data (transformed thanks to Nilearn)(the result is called ridge-indiv).

### MEG

TODO


## File organization

 
Folders within the project main directory  (`$LPP`) are organized in the following way:

<pre>
├── <b>paradigm</b> <i>(code to run the experiment)</i>
├── <b>code</b> <i>(all the code of all the analysis)</i>
│   ├── <b>MEG</b> <i>(code of the MEG analysis pipeline)</i>
│   ├── <b>fMRI</b> <i>(code of the fMRI analysis pipeline)</i>
│   ├── <b>models</b> <i>(code related to models initialization/training/generation)</i>
│   │   ├── <b>english</b>
│   │   │   ├── <b>LSTM</b> <i>(LSTM framework)</i>
│   │   │   ├── <b>RMS</b> <i>(Framework for wave properties analysis)</i>
│   │   │   ├── <b>WORDRATE</b> <i>(Framework for simple linguistics properties analysis)</i>
│   │   │   ├── lstm_wikikristina_embedding-size_200_nhid_300_nlayers_1_dropout_01.py <i>(instantiation of a LSTM model)</i>
│   │   │   ├── lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01.py <i>(instantiation of a LSTM model)</i>
│   │   │   └── ...
│   │   └── <b>french</b>
│   └── <b>utilities</b> <i>(utilities functions: parameters settings, splitter for CV, ...)</i>
├── <b>data</b> <i>(all the raw data acquired from sources)</i>
│   ├── <b>fMRI</b> <i>(fMRI data, 9 runs per subject)</i>
│   │   └── <b>english</b>
│   │       └── <b>sub-057</b>
│   │           └── <b>func</b>
│   ├── <b>wave</b> <i>(wave files data, 9 runs, data for models training)</i>
│   │   ├── <b>english</b>
│   │   └── <b>french</b>
│   └── <b>text</b> <i>(text data, raw text, division in 9 runs, onsets/offsets for each runs, data for models training)</i>
│       ├── <b>english</b>
│       │   ├── <b>lstm_training</b>
│       │   └── <b>onsets-offsets</b>
│       └── <b>french</b>
└── <b>derivatives</b> <i>(results of the code above)</i>
    ├── <b>MEG</b>
    └── <b>fMRI</b> <i>(results from the fMRI pipeline in code/fMRI/)</i>
        ├── <b>design-matrices</b> <i>(concatenation of features associated with different models of interest)</i>
        │   └── <b>english</b>
        ├── <b>features</b> <i>(Raw-features convolved with an 'hrf' kernel)</i>
        │   └── <b>english</b>
        ├── <b>glm-indiv</b> <i>(GLM model fitted on fMRI data with a design-matrix)</i>
        │   └── <b>english</b>
        ├── <b>models</b> <i>(trained models)</i>
        │   └── <b>english</b>
        ├── <b>raw_features</b> <i>(Result of a model generation from the text/wave file of LPP, concatenated with the adequate onsets file)</i>
        │   └── <b>english</b>
        └── <b>ridge-indiv</b> <i>(Ridge model fitted on fMRI data with a design-matrix)</i>
            └── <b>english</b>
</pre>



The script `create_architecture.py` automatically generates this tree structure (but does not fill it).



- **code**

1. Generation of raw-features from the text (or audio) of "Le Petit Prince" thanks to the selected models.
2. Concatenation of the raw-feature dataframe with an onset file (the result is called raw-features).
3. Convolution of the newly constructed dataframe with an 'hrf' kernel (the result is called features).
4. Construction of a design-matrix by concatenation of the features associated with the different models of interest (the result is called design-matrix).
5. Ridge (cross validated) regression between our design-matrix and the fMRI data (transformed thanks to Nilearn)(the result is called ridge-indiv**.

### MEG ###


**TODO**




2. **data**
    - we have the fMRI data organized following the BIDS standard except for the name of the final file
    - the MEG should be added in a few months
    - there is the text of LPP, the text divided into 9 runs, the original onset-offsets of LPP and training data for models
    - wave files, meaning the content of the audio book with the textgrid files and training data for models

3. **derivatives**
    - MEG
    - fMRI (every script of `code/fMRI/` fills a folder of the same name here, the same goes for `code/models/`)



## Running the scripts ##


To train a given model,  you need to create a module `model_name.py` in `$LPP/code/models/language/`, with the following functions:
- `load`: that returns the trained model 
- `generate`: that take as arguments a model, a path to the input run, a language and a textgrid dataframe and generate raw-features

And add at the end of the script:

```
if __name__=='__main__':
    train(model)
```


### Model training ###


###  Training a model

To train *model_name.py* in a given *language*, just write:

```
cd $LPP
cd code
python models/language/model_name.py
```

### fMRI pipeline ###

<<<<<<< HEAD
### Creating a new analysis

    1. Create a new directory in TODO and copy `code/utilities/settings_template.py` there.
 
    2. If necessary add new data/raw features in TODO

    3. Create a link from code/utilities/settings.py to the setting.py you want 

---
cd $LPP
cd code/fMRI
doit
---
  
`doit` behaves like the make utility: it will only recompute the necessary files
    based on their timestamps.  If you still want to recreate all the files, run:

---
cd $LPP
cd code/fMRI
doit clean
doit forget
---

### Results

Available tools:
    

- scatter plot comparison of r2 distributions per ROI in the brain for 2 given models

To run such an analysis, you should first fill in the `analysis.yaml` file with the name of the model you want to study and the name of the study that this scatter plot is suppose to enlighten (e.g. syntax VS semantic).
Then run the following command line:



     cd $LPP/code/fMRI
     python analysis.py $LPP/code/analysis.yaml

