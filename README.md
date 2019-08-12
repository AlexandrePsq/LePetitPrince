# LePetitPrince

This repository includes the code of "Le Petit Prince" project.
(LPP = Le Petit Prince)

English, French and Chinese participants were scanned using fMRI while listening to the whole audiobook of the Little Prince (~90min), in their native language.

The audio stream was segmented into nine parts presented in 9 successive runs:

- Chapters 1 to 3 --> run 1
- Chapters 4 to 6 --> run 2
- Chapters 7 to 9 --> run 3
- Chapters 10 to 12 --> run 4
- Chapters 13 to 14 --> run 5
- Chapters 15 to 19 --> run 6
- Chapters 20 to 22 --> run 7
- Chapters 23 to 25 --> run 8
- Chapters 26 to 27 --> run 9


## Data acquisition ##

- English Multiecho 

- French Multiecho Siemens 3T

- Chinese


## Methodology ##

### fMRI ###

Selection and implementation of different Language Models.

Analysis pipeline:

1. Generation of raw-features from the text (or audio) of "Le Petit Prince" thanks to the selected models.
2. Concatenation of the raw-feature dataframe with an onset file (the result is called raw-features).
3. Convolution of the newly constructed dataframe with an 'hrf' kernel (the result is called features).
4. Construction of a design-matrix by concatenation of the features associated with the different models of interest (the result is called design-matrix).
5. Ridge (cross validated) regression between our design-matrix and the fMRI data (transformed thanks to Nilearn)(the result is called ridge-indiv**.

### MEG ###


**TODO**



## Data architecture ##

The files are organized in the following overall folder structure:

---
    paradigm (experiences information, stimuli)
    oldstuff (oldscripts, personnal data/code, ...)
    code (all the code of all the analysis)
        MEG (code of the MEG analysis pipeline)
        fMRI (code of the fMRI analysis pipeline)
        models (code related to models initialization/training/generation)
            english
                LSTM (LSTM framework)
                RMS (Framework for wave properties analysis)
                WORDRATE (Framework for simple linguistics properties analysis)
                lstm_wikikristina_embedding-size_200_nhid_300_nlayers_1_dropout_01.py (instantiation of a LSTM model)
                lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01.py (instantiation of a LSTM model)
                ...
            french
        utilities (utilities functions: parameters settings, splitter for CV, ...)
    data (all the raw data acquired from sources)
        fMRI (fMRI data, 9 runs per subject)
            english
                sub-057
                    func
        wave (wave files data, 9 runs, data for models training)
            english
            french
        text (text data, raw text, division in 9 runs, onsets/offsets for each runs, data for models training)
            english
                lstm_training
                onsets-offsets
            french
    derivatives (results of the code above)
        MEG
        fMRI (results from the fMRI pipeline in code/fMRI/)
            design-matrices (concatenation of features associated with different models of interest)
                english
            features (Raw-features convolved with an 'hrf' kernel)
                english
            glm-indiv (GLM model fitted on fMRI data with a design-matrix)
                english
            models (trained models)
                english
            raw_features (Result of a model generation from the text/wave file of LPP, concatenated with the adequate onsets file)
                english
            ridge-indiv (Ridge model fitted on fMRI data with a design-matrix)
                english
---

The script `create_architecture.py` automatically generates this tree structure (but does not fill it).


The three main parts of the project are:

1. **code**
    - MEG data analysis pipeline
    - fMRI data analysis pipeline
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
    - models
        - \*XXXX\* : framework associated with a kind of model (e.g. LSTM)*
            - model.py *(class definition)*
            - train.py *(train the model)*
            - tokenizer.py *(adequate tokenizer for the model training and generation)*
            - utils.py *(utilities functions)*
        - \*xxxx\* : instantiation of a given class model

2. **data**
    - we have the fMRI data organized following the BIDS standard except for the name of the final file
    - the MEG should be added in a few months
    - there is the text of LPP, the text divided into 9 runs, the original onset-offsets of LPP and training data for models
    - wave files, meaning the content of the audio book with the textgrid files and training data for models

3. **derivatives**
    - MEG
    - fMRI (every script of `code/fMRI/` fills a folder of the same name here, the same goes for `code/models/`)



## Running the scripts ##

To train a given model called *model_name.py* in a given *language* and use it in the pipeline, create a module `model_name.py` in `LPP/code/models/language/`, and define in it the functions:
- `load`: that returns the trained model 
- `generate`: that take as arguments a model, a path to the input run, a language and a textgrid dataframe and generate raw-features

Then, add at the end of the script:

    if __name__=='__main__':
        train(model)

### Model training ###

To train a given model `model_name.py` in a given language:

    export LPP=<path to the root of the project>
    cd $LPP/code
    python models/language/model_name.py


### fMRI pipeline ###

The parameters for the fMRI analysis pipeline are in `code/utilities/settings.py`.

To run the pipeline:

    cd $LPP/code/fMRI
    doit

`dodo.py` will not run a file that has already been run except if it has been modified.
If you still want to do so, run:


    cd $LPP/code/fMRI
    doit clean
    doit forget


Running `doit` will first create the adequate architecture and then start the fMRI pipeline.


### Analysis ###

Available analysis so far:
- scatter plot comparison of r2 distributions per ROI in the brain for 2 given models

To run such an analysis, you should first fill in the `analysis.yaml` file with the name of the model you want to study and the name of the study that this scatter plot is suppose to enlighten (e.g. syntax VS semantic).
Then run the following command line:


     cd $LPP/code/fMRI
     python analysis.py $LPP/code/analysis.yaml
