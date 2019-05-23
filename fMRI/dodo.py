import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Rois, Paths
from utilities.utils import get_output_parent_folder, get_path2output
from itertools import combinations, product
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)
warnings.simplefilter(action='ignore', category=ImportWarning)


from os.path import join

############################################################################
################################# SETTINGS #################################
############################################################################


## Language
language = 'en' # languages = ['en', 'fr']

## List of all the subjects
subjects = Subjects() # for subject in subjects.subject_lists[language]:

## FMRI sampling period
tr = 2

## Number of runs
nb_runs = 3
run_names = ['run{}'.format(i) for i in range(1,nb_runs + 1)]

## Models
# models = sorted(['test-1', 'test-5', 'test-10', 'test-20', 'test-50', 'test-100'])
models = sorted(['test-1'])

## Aggregated models (for design matrices contruction)
# aggregated_models = [' '.join(item) for i in range(1, len(models)+1) for item in combinations(models, i)]
aggregated_models = models

## Testing
test = True 

## Overwritting
overwrite = False

## Parallelising
parallel = True

## Alphas list for voxel-wised analysis
alphas = np.logspace(-3, -1, 30)


optional = ''
optional += '--overwrite ' if overwrite else ''
optional_parallel = '--parallel ' if parallel else ''


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
            extension_input = '.wav'
        else:
            input_data_type = 'text'
            extension_input = '.txt'
        input_parent_folder = join(paths.path2data, '{0}/{1}/test'.format(input_data_type, language))
        dependencies = [join(input_parent_folder, '{0}_{1}_{2}'.format(input_data_type, language, run_name) + extension_input) for run_name in run_names]
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, model)
        targets = [get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension) for run_name in run_names]
        yield {
            'name': model,
            'file_dep': ['raw_features.py'] + dependencies,
            'targets': targets,
            'actions': ['python raw_features.py --language {} --model {} '.format(language, model) + optional + optional_parallel],
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
            'actions': ['python features.py --tr {} --language {} --model {} '.format(tr, language, model) + optional + optional_parallel],
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
    extension = '.csv'
    subjects = Subjects()

    for models in aggregated_models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
        dependencies = [get_path2output(input_parent_folder, input_data_type, language, models, run_name, extension) for run_name in run_names]
        targets = [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.nii.gz') for subject in subjects.get_all(language, test)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.png') for subject in subjects.get_all(language, test)]
        yield {
            'name': models,
            'file_dep': ['glm-indiv.py'] + dependencies,
            'targets': targets,
            'actions': ['python glm-indiv.py --language {} --model_name {} '.format(language, models) + optional + optional_parallel + ' --subjects ' + ' '.join(subject for subject in subjects.get_all(language, test))],
        }


def task_ridge_indiv():
    """Step 4 bis: Generate r2 & alphas maps (if voxel wised enabled) from design matrices and fMRI data in a given language for a given model."""
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'ridge-indiv'
    extension = '.csv'
    subjects = Subjects()

    for models in aggregated_models:
        output_parent_folder = get_output_parent_folder(source, output_data_type, language, models)
        input_parent_folder = get_output_parent_folder(source, input_data_type, language, models)
        dependencies = [get_path2output(input_parent_folder, input_data_type, language, models, run_name, extension) for run_name in run_names]
        targets = [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.nii.gz') for subject in subjects.get_all(language, test)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'r2_test', subject)+'.png') for subject in subjects.get_all(language, test)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'alphas', subject)+'.nii.gz') for subject in subjects.get_all(language, test)] \
                + [join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(output_data_type, language, models, 'alphas', subject)+'.png') for subject in subjects.get_all(language, test)]
        yield {
            'name': models,
            'file_dep': ['ridge-indiv.py'] + dependencies,
            'targets': targets,
            'actions': ['python ridge-indiv.py --language {} --model_name {} --voxel_wised '.format(language, models) \
                + optional + optional_parallel \
                    + ' --subjects ' + ' '.join(subject for subject in subjects.get_all(language, test))\
                        + ' --alphas ' + ' '.join(str(alpha) for alpha in alphas)]
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
