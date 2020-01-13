import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output, check_folder
from itertools import combinations, product
import yaml
import glob
import numpy as np

import warnings
warnings.simplefilter(action='ignore')


from os.path import join

############################################################################
################################# SETTINGS #################################
############################################################################
### Here, are retrieved the parameters from the $LePetitPrince/code/utilities/settings.py
### file.
### For any parameter modification, modify the source file indicated just above.


params = Params()
paths = Paths()
subjects = Subjects()

## Set parameters
path2models = os.path.join(paths.path2code, 'fMRI', 'models.yml')
path2parameters = os.path.join(paths.path2code, 'utilities', 'parameters.yml')
alphas = np.logspace(-3, 3, 30)
with open(path2models, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except :
        print('Error when loading models.yml...')
        quit()
with open(path2parameters, 'r') as stream:
    try:
        parameters = yaml.safe_load(stream)
    except :
        print('Error when loading parameters.yml...')
        quit()
run_names = ['run{}'.format(i) for i in range(1, parameters['nb_runs'] + 1)]

optional_parallel = '--parallel ' if parameters['parallel'] else ''


############################################################################
################################# PIPELINE #################################
############################################################################
### Here is defined the pipeline architecture, where each function (starting 
### with 'task_') represents one step of the pipeline.
### Each step of the pipeline will be call the as many times as there is 
### elements in the 'actions' key of the dictionary that is yielded.


paths = Paths()

def task_features():
    """Generate features (=fMRI regressors) from predictions 
    (csv file with 3 columns onset-amplitude-duration) by convolution 
    with an hrf kernel."""
    input_data_type = 'predictions'
    output_data_type = 'features'
    extension = '.csv'
    source = 'fMRI'
    for model in data['models']:
        model_name = model['model_name']
        language = model['language']
        onsets_paths = ' '.join(sorted(glob.glob(os.path.join(paths.path2data, 'wave', language, 'onsets-offsets', model['onset_type'] + '_run*.csv'))))
        dependencies = [get_path2output(source=source, data_type=input_data_type, language=language, model=model_name, run_name=run_name, extension=extension) for run_name in run_names]
        targets = [get_path2output(source=source, data_type=output_data_type, language=language, model=model_name, run_name=run_name, extension=extension) for run_name in run_names]
        optional = ''
        optional += '--overwrite ' if model['overwrite'] else ''
        optional += '--shift_surprisal ' if model['shift_surprisal'] else ''
        yield {
            'name': model['surname'],
            'file_dep': ['features.py'] + dependencies,
            'targets': targets,
            'actions': ['python features.py --tr {} --language {} --model {} --onsets_paths {} '.format(parameters['tr'], language, model_name, onsets_paths) + optional + optional_parallel],
        }



def task_generate_command_lines():
    """Generate command lines to run the ridge regression
    on the nodes of a cluster. The ridge regression generate cross-validated
    R2 maps."""
    source = 'fMRI'
    input_data_type = 'features'
    extension = '.csv'

    all_dependencies = []

    language = data['aggregated_models']['language']
    subject_argument = ' '.join(subjects.get_all(language))
    optional = '--overwrite ' if data['aggregated_models']['overwrite'] else ''
    jobs_state_folder = os.path.join(paths.path2root, 'command_lines/jobs_states')
    check_folder(jobs_state_folder)


    for key in data['aggregated_models']['all_models'].keys():
        for model in data['aggregated_models']['all_models'][key]:
            model_name = model['model_name']
            all_dependencies += [get_path2output(source=source, data_type=input_data_type, language=language, model=model_name, run_name=run_name, extension=extension) for run_name in run_names]

    yield {
        'name': 'Generating command lines...',
        'file_dep': [os.path.join(paths.path2code, 'fMRI/check_computations.py')] + all_dependencies,
        'targets': [],
        'actions': ['python check_computations.py --jobs_state_folder {} \
                                                    --subjects {} \
                                                    --path2models {} \
                                                    --path2parameters {} '.format(jobs_state_folder, subject_argument, path2models, path2parameters) + optional ]

                                                            }
