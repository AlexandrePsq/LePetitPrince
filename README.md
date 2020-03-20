# LePetitPrince

Code of the "Le Petit Prince" project.

English, French and Chinese participants were scanned using fMRI while listening to the whole audiobook of the Little Prince (~90min), in their native language.

Natural Language Processing models are used to generate predictors (lexical frequency, syntactic complexity, semantic features, etc.) that then used to predict brain signals.

The audiobook was segmented into nine audiofiles, presented in successive runs:


The wave files (`$LPP/data/wave/{english,french}/*.wav`) were annotated for words onset-onsets (`$LPP/data/wave/{french,english}/*.TextGrid`)

## fMRI data acquisition

TODO: add the description of the sequences in English and in French

- English Multiecho:  `$LPP/data/fMRI/english`
- French Multiecho Siemens 3T `$LPP/data/fMRI/french`
- Chinese


## File organization

 Folders within the project main directory  (`$LPP`) are organized in the following way:

<pre>
├── <b>paradigm</b> <i>(code to run the experiment)</i>
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


The script `create_architecture.py` automatically generates this tree structure (but does not fill it).



### Model training ###

TODO: explain what 'model training mean' means (I guess going from a corpus to a network weights (?)) 

To train *model_name.py* in a given *language*, just write:

```
cd $LPP
cd code
python models/language/model_name.py
```

The three main parts of the project:

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
>>>>>>> 627b567b743d57d361cb6c2d8b1832b20a7b1475

### fMRI pipeline ###

### Creating a new analysis

    1. Create a new directory in TODO and copy `code/utilities/settings_template.py` there.
 
    2. If necessary add new data/raw features: TODO

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
=======
To run the fMRI pipeline, first fill a yaml template specifying the parameters of your analysis and call `main.py`.
>>>>>>> 627b567b743d57d361cb6c2d8b1832b20a7b1475

Available tools:
    

### Analysis


#### TO BE MODIFIED
Available analysis so far:

    - scatter plot comparison of r2 distributions per ROI in the brain for 2 given models

To run such an analysis, you should first fill in the `analysis.yaml` file with the name of the model you want to study and the name of the study that this scatter plot is suppose to enlighten (e.g. syntax VS semantic).
Then run the following command line:


     cd $LPP/code/fMRI
     python analysis.py $LPP/code/analysis.yaml

