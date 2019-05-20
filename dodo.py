from utilities.settings import Subjects, Rois, Paths
from utilities.utils import *
from itertools import combinations, product

from os.path import join

############################################################################
################################# SETTINGS #################################
############################################################################


# Language
language = 'en' # languages = ['en', 'fr']

# List of all the subjects
subjects = Subjects() # for subject in subjects.subject_lists[language]:

# FMRI sampling period
tr = None

# Number of runs
nb_runs = 9
run_names = ['run{}'.format(i) for i in range(1,10)]

# Models
models = sorted([])

# Aggregated models (for design matrices contruction)
aggregated_models = [' '.join(item) for i in range(1, len(models)) for item in combinations(models, i)]

# Testing
test = False 

# Overwritting
overwrite = False

# Parallelising
parallel = True

optional = '--test ' if test else ''
optional += '--overwrite ' if overwrite else ''
optional += '--parallel ' if parallel else ''


############################################################################
################################# PIPELINE #################################
############################################################################

paths = Paths()


def task_raw_features():
    """Step 1: Generate raw features from raw data (text, wave) model predictions."""
    extension = '.csv'
    output_data_type = 'raw_features'
    source = 'fMRI'
    
    for model in models:
        if model in ['rms', 'f0']:
            input_data_type = 'wave'
        else:
            input_data_type = 'text'
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, model)
        targets = [get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension) for run_name in run_names]
        yield {
            'name': model,
            'file_dep': ['raw_features.py'],
            'targets': targets,
            'actions': ['python raw_features.py --language {} --model {} '.format(language, model) + optional],
        }


def task_features():
    """Step 2: Generate features (=fMRI regressors) from raw_features (csv file with 3 columns onset-amplitude-duration) by convolution with an hrf kernel."""
    input_data_type = 'raw_features'
    output_data_type = 'features'
    extension = '.csv'
    source = 'fMRI'

    for model in models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, model)
        input_parent_folder = get_output_parent_folder(source, input_data_type, language, model)
        dependencies = [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
        targets = [get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension) for run_name in run_names]
        yield {
            'name': model,
            'file_dep': ['features.py'] + dependencies,
            'targets': targets,
            'actions': ['python features.py --language {} --model {} '.format(language, model) + optional],
        }


def task_design_matrices():
    """Step 3: Generate design matrices from features in a given language."""
    input_data_type = 'features'
    output_data_type = 'design-matrices'
    extension = '.csv'
    source = 'fMRI'

    for models in aggregated_models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
        dependencies = []
        for model in models.split():
            input_parent_folder = get_output_parent_folder(source, input_data_type, language, model)
            dependencies += [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
        targets = [get_path2output(output_parent_folder, output_data_type, language, models, run_name, extension) for run_name in run_names]
        yield {
            'name': models,
            'file_dep': ['design-matrices.py'] + dependencies,
            'targets': targets,
            'actions': ['python design-matrices.py --language {} --models {} '.format(language, models) + optional],
        }


def task_glm_indiv():
    """Step 4: Generate r2 maps from design matrices and fMRI data in a given language for a given model."""
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'glm-indiv'

    for models in aggregated_models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
        dependencies = [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
        targets = [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.nii.gz') for subject in subjects.get_all(language)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.png') for subject in subjects.get_all(language)]
        yield {
            'name': models,
            'file_dep': ['glm-indiv.py'] + dependencies,
            'targets': targets,
            'actions': ['python glm-indiv.py --language {} --model_name {} '.format(language, models) + optional],
        }


def task_ridge_indiv():
    """Step 4 bis: Generate r2 & alphas maps (if voxel wised enabled) from design matrices and fMRI data in a given language for a given model."""
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'ridge-indiv'

    for models in aggregated_models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
        dependencies = [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
        targets = [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.nii.gz') for subject in subjects.get_all(language)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.png') for subject in subjects.get_all(language)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'alphas', subject)+'.nii.gz') for subject in subjects.get_all(language)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'alphas', subject)+'.png') for subject in subjects.get_all(language)]
        yield {
            'name': models,
            'file_dep': ['ridge-indiv.py'] + dependencies,
            'targets': targets,
            'actions': ['python ridge-indiv.py --language {} --model_name {} '.format(language, models) + optional],
        }


#def task_glm_group():
#    """Step 5: Compute GLM group analysis."""
#    source = 'fMRI'
#    input_data_type = 'glm-indiv'
#    output_data_type = 'glm-group'
#
#    for models in aggregated_models:
#        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
#        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
#        dependencies = [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
#        targets = [get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension) for run_name in run_names]
#        yield {
#            'name': models,
#            'file_dep': ['glm-group.py'] + dependencies,
#            'targets': targets,
#            'actions': ['python glm-group.py --language {} --model_name {} '.format(language, models) + optional],
#        }


#def task_ridge_group():
#    """Step 5 bis: Compute Ridge group analysis."""
#    source = 'fMRI'
#    input_data_type = 'ridge-indiv'
#    output_data_type = 'ridge-group'
#
#    for models in aggregated_models:
#        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
#        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
#        dependencies = [get_path2output(input_parent_folder, input_data_type, language, model, run_name, extension) for run_name in run_names]
#        targets = [get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension) for run_name in run_names]
#        yield {
#            'name': models,
#            'file_dep': ['ridge-group.py'] + dependencies,
#            'targets': targets,
#            'actions': ['python ridge-group.py --language {} --model_name {} '.format(language, models) + optional],
#        }



############################################################################
############################################################################
############################################################################
