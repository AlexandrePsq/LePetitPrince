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



This huge dataset will be shared through neurovault.


## Methodology  ##

We abode by the following methodology:

### fMRI ###

Selection and implementation of different Language Models (GloVe, LSTM, BERT/CamemBERT, GPT-2).

Analysis pipeline:

0.  Generation of deep-representations from the text (or audio) of "Le Petit Prince" thanks to the selected models.
1.  Construction of a design-matrix by concatenation of the representations associated with the different models of interest.
2.  Nested cross-validated Ridge-regression between our design-matrix and the fMRI data (transformed thanks to Nilearn).

The nested cross-validation (NCV) is composed of a main cross-validation (CV) over the R2 values, and for each split of the main 
cross-validation, another cross-validation is done to determined the regularization (hyperparameter) of the ridge-regression.
If needed (and before the regression), a compression of the representations is done during the inner/outer CV.
We then concatenate the dataframe of compressed representations with an onset file and convolve the newly constructed dataframe 
with an 'hrf' kernel to get the regressors that will be aligned to the fMRI data thanks to the regression.

### MEG ###

**TODO**



## Data architecture ##

The files are organized in the following overall folder structure:

<pre>
 ---
├── <b>paradigm</b> <i>(experiences information, stimuli)</i>
├── <b>oldstuff</b> <i>(oldscripts, personnal data/code, ...)</i>
├── <b>code</b> <i>(all the code of all the analysis)</i>
│       ├── <b>MEG</b> <i>(code of the MEG analysis pipeline)</i>
│       └── <b>fMRI</b> <i>(code of the fMRI analysis pipeline)</i>
│               ├── <b>data_compression.py</b> <i>(Class regrouping methods to compress the representation data)</i>
│               ├── <b>data_transformation.py</b> <i>(Class regrouping methods to transform the data: standardization, creating rergessors, ...)</i>
│               ├── <b>encoding_models.py</b> <i>(Class where the Linear (regularized or not) model is implemented)</i>
│               ├── <b>logger.py</b> <i>(Logging class to check piepeline status)</i>
│               ├── <b>main.py</b> <i>(Launch the pipeline for the given yaml config file)</i>
│               ├── <b>regression_pipeline.py</b> <i>(Class implementing the pipeline for the regression analysis)</i>
│               ├── <b>requirements.txt</b> <i>(required librairies + versions)</i>
│               ├── <b>splitter.py</b> <i>(Class regrouping splitting/distributing methods)</i>
│               ├── <b>task.py</b> <i>(Class implementing a Task which is a step of the pipeline)</i>
│               ├── <b>template.yml</b> <i>(Yaml config file to fill for each call of main.py)</i>
│               └── <b>utils.py</b> <i>(utilities functions: parameters settings, fetching, reading/writing ...)</i>
├── <b>data</b> <i>(all the raw data acquired from sources)</i>
│       └── <b>fMRI</b> <i>(fMRI data, 9 runs per subject)</i>
│               └── <b>english</b>
│                       └── <b>sub-057</b>
│                               └── <b>func</b>
└── <b>derivatives</b> <i>(results of the code above)</i>
      ├── <b>MEG</b>
      └── <b>fMRI</b> <i>(results from the fMRI pipeline in code/fMRI/)</i>
              ├── <b>deep-representations</b> <i>(deep representation dataframes extracted from the models activity)</i>
              │  ├── <b>english</b>
              │  │  └── <b> model_of_interest </b>
              │  │         ├── <b> deep_representation_run_1.csv </b>
              │  │         ├──  ...
              │  │         └── <b> deep_representation_run_<i>n</i>.csv </b>
              │  └── <b>french</b>
              └── <b>maps</b> <i>Maps deriving from our pipeline</i>
                    ├── <b>english</b>
                    │       └── <b>sub-057</b>
                    │               └── <b> model_of_interest </b>
                    │                           ├── R2.nii.gz
                    │                           └── R2.png
                    └── <b>french</b>
</pre>

To give more insights on the three main parts of the project:

- **code**
    - MEG data analysis pipeline
    - fMRI data analysis pipeline
        - splitter.py *(Split/distribute the data)*
        - data_compression.py *(Compress data representations if needed: PCA, etc...)*
        - data_transformation.py *(Transform data representations: create regressor by convolving with an HRF kernel and standardize before regression)*
        - encoding_models.py *((Regularized) Linear model that fit the regressors to fmri data)*
        - task.py *(Step of the pipeline to be executed)*
        - regression_pipeline.py *(Define and execute the pipeline)*
        - requirements.txt *(Required libraries)*
        - logger.py *(Report pipeline progression)*
        - main.py *(Execute pipeline with the config from the yaml file)*
        - template.yml *(Yaml file specifying the configuration of the analysis)*
        - utils.py *(Utilities functions)*

- **data**
    - we have the fMRI data organized following the BIDS standard except for the name of the final file
    - the MEG should be added in a few months
    - there is the text of LPP, the text is divided into 9 runs, the original onset-offsets of LPP and training data for models

- **derivatives**
    - MEG
    - fMRI (to prevent useless use of memory, we only save deep-model representations as well as the final maps output)



## Executing scripts ##

### fMRI pipeline ###

Protocol:
- Upload your deep-representation matrices as a .csv file in <pre>$LPP/derivatives/fMRI/deep-representations/<i>language</i>/<i>model_name</i>/</pre>.
- Be careful that each '.csv' filename contain its run number.
- Be sure to have the adequate onsets/offsets files for each model included in your analysis (the path should be specified in the yaml template).
- Optional: if the creation of a regressor for a given model require specific *duration*, create the adequate array for each run (the path should be specified in the yaml template), otherwise a vector of *1* should be used.
- Check the *requirements.txt* to see if you have all the libraries needed.
- Run the following command:
<pre>python main.py --yaml_file <i>path_to_yaml_file</i> --input <i>path_to_representations_folder</i> --output <i>path_to_output_folder</i> --logs <i>path_to_log_file</i></pre>


### Analysis ###


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
