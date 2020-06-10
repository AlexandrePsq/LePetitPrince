# Code to compute cross-validated R² or r maps from model predictions and fMRI data (and MEG???)

Given

- a sequence of stimuli, 
- fMRI data from subjects exposed to these sequence
- one or several models making quantitative predictions from each stimulus in  the sequence

The code allows you build a flexible analysis pipeline combining:

- data compression methods (already coded or that you can add),
- data transformation methods (standardization, convolution with a kernel, or whatever your heart desires...),
- splitting strategies,
- encoding models,
- or any task that you might find useful (and that you are willing to code). <-???
- functions that allow you to generate R² or r maps.


For example,  the pipeline programmed in `main.py` fits stimuli-representations of several lanuage models to fMRI data obtained from participants listening to an audiobook (in 9 runs). R² are computed from a nested-cross validated Ridge-regression.

The pipeline runs for tuples (1 subject, 1 model), to facilitate distributing the code on a cluster.



### Setup ###


0. Check `requirements.txt` for the list of pyhton modules required.

1. For each model to be evaluated, generate the "stimuli-representations" from the sequence of stimuli.

  A "stimuli representation" is a set of "features", that is, numeric vectors of the same length as the sequence. It must be saved as a matrix in a `.csv` file in <pre>$LPP/derivatives/fMRI/stimuli-representations/<i>{language}</i>/<i>model_name</i>/</pre>, with one file per run.

3. Upload the adequate onsets/offsets files of the stimuli for each model included in your analysis (the path to onsets/offsets folder, and the type of each onset/offsets for each model should be specified in the `yaml` template). <-- yaml template has not been mentioned yet ??? 

4. Optional: if the creation of a regressor for a given model requires specific *duration*, you need to create the adequate array for each run (the path should be specified in the yaml template), otherwise a vector of *1* should be used.



### Structure of main.py ###

0. Define variables.
1. Fetch (or compute) a global masker (+ a smoothed version) over all subjects (to have a common analysis ground).
2. Retrieve the arguments for classes instanciations.
3. Instanciate the classes.
4. Define dependencies relations between classes.
5. Load paths to stimuli-representation and fMRI data.
6. Extract the needed columns of the stimuli-representations, and concatenate them.
7. Load fMRI data and add small variation* to voxel with constant activation (otherwise they generate error with the Pearson correlation coefficient).
8. Fit the pipeline.
9. Launch the pipeline.
10. Create brain maps from the R2 (Pearson coefficient) maps resulting from the pipeline.

*small variation is added to the first time step only.

### Pipeline steps: ###


0. We split the data into 9 splits for a CV over the R2.
    1. We split each train set of each former split for a CV over the regularization hyperparameter alpha of our encoding model.
    2. For each split of the inner CV, we compress the stimuli-representation of the models that require it.
    3. For each split of the inner CV, we compute the regressor by convolving the (compressed or not) stimuli-representations with an hrf kernel.
    4. For each split of the inner CV, we standardize the newly computed regressors before the Ridge regression.
    5. For each split of the inner CV, we fit Ridge encoding models for each voxel and each alpha included in our grid search, and compute R2/Pearson values.
1. For each split of the outter CV, we compress the stimuli-representations of the models that require it.
2. For each split of the outter CV, we compute the regressor by convolving the (compressed or not) stimuli-representations with an hrf kernel.
3. For each split of the outter CV, we standardize the newly computed regressors before the Ridge regression.
4. For each split of the outter CV, we fit Ridge encoding models with the best alpha* for each voxel and compute R2/Pearson values.

*the best alpha is determined for each voxel.



## Executing scripts ##

### fMRI pipeline ###

After having verified the requirements, run the following command:
<pre>python main.py --yaml_file <i>path_to_yaml_file</i> --input <i>path_to_representations_folder</i> --output <i>path_to_output_folder</i> --logs <i>path_to_log_file</i></pre>




## Data architecture ##

The files are organized in the following overall folder structure (`{language}` can take the values `french` or `english`)

<pre>
 ---
├── <b>paradigm</b> <i>(scripts to run the experiment and stimuli)</i>
├── <b>oldstuff</b> <i>(older scripts, personnal data/code, ...)</i>
├── <b>code</b> <i>(complete code for analyses)</i>
│   ├── <b>MEG</b> <i>(code for MEG analysis pipeline)</i>
│   └── <b>fMRI</b> <i>(code for fMRI analysis pipeline)</i>
│       ├── data_compression.py <i>(Class regrouping methods to compress the representation data)</i>
│       ├── data_transformation.py <i>(Class regrouping methods to transform the data: standardization, creating rergessors, ...)</i>
│       ├── encoding_models.py <i>(Class where the Linear (regularized or not) model is implemented)</i>
│       ├── logger.py <i>(Logging class to check piepeline status)</i>
│       ├── main.py <i>(Launch the pipeline for the given yaml config file)</i>
│       ├── regression_pipeline.py <i>(Class implementing the pipeline for the regression analysis)</i>
│       ├── requirements.txt <i>(required librairies + versions)</i>
│       ├── splitter.py <i>(Class regrouping splitting/distributing methods)</i>
│       ├── task.py <i>(Class implementing a Task which is a step of the pipeline)</i>
│       ├── template.yml <i>(Yaml config file to fill for each call of main.py)</i>
│       └── utils.py <i>(utilities functions: parameters settings, fetching, reading/writing ...)</i>
├── <b>data</b> <i>(all the raw data acquired from sources)</i>
│   ├── <b>fMRI</b> <i>(fMRI data, 9 runs per subject)</i>
│   │   └── <b><i>{language}</i></b>
│   │       └── <b>sub-057</b>
│   │           └── <b>func</b>
│   ├── <b>text</b> <i>(text of each chapter)</i>
│   │   └── <b><i>{language}</i></b>
│   │       └── <b>text_{language}_run[1-9].txt</b>
│   ├── <b>onsets-offsets</b> <i>(onsets-offsets data)</i>
│   │   └── <b><i>{language}</i></b>
│   │       ├── <b>word_run_1.csv</b>
│   │       ├──  ...
│   │       ├── <b>word_run_<i>n</i>.csv</b>
│   │       ├── <b>word+punctuation_run_1.csv</b>
│   │       ├──  ...
│   │       └── <b>word+punctuation_run_<i>n</i>.csv</b>
│   └── <b>stimuli-representations</b> <i>(stimuli representation dataframes extracted from the models activity)</i>
│       └── <b><i>{language}</i></b>
│           └── <b><i>model_of_interest</i> </b>
│               ├── deep_representation_run_1.csv 
│               ├──  ...
│               └── deep_representation_run_<i>n</i>.csv
│
└── <b>derivatives</b> <i>(results of the code above)</i>
    └── <b>MEG</b>
        └── <b>fMRI</b> <i>(results from the fMRI pipeline in code/fMRI/)</i>
            └── <b>maps</b> (<i>Maps deriving from our pipeline</i>)
                └── <b><i>{language}</i></b>
                    └── <b>sub-057</b>
                        └── <b> <i>model_of_interest</i> </b>
                            ├── R2.nii.gz
                            └── R2.png
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



# Complementary information on the initial project: Le Petit Prince

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





