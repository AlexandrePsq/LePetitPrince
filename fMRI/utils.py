import os
import yaml
import inspect
import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker


#########################################
############ Basic functions ############
#########################################

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
            quit()
    return parameters
    
def write(path, text, end='\n'):
    """Write in the specified text file."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)

def save(object_to_save, path):
    if isinstance(object_to_save, np.ndarray):
        extension = '.npy'
        np.save(path+extension, object_to_save)
    elif isinstance(object_to_save, pd.DataFrame):
        extension = '.csv'
        object_to_save.to_csv(path+extension, index=False)
    # others to add ?

def merge_dict(list_of_dict):
    """ Merge a list of dictionaries into a single dictionary.
    Arguments:
        - list_of_dict: list (of dicts)
    """
    result = {key: value for d in list_of_dict for key, value in d.items()}
    return result

def clean_nan_rows(array):
    """ Remove rows filled with NaN values.
    Iterate row by row to keep the array structure (no flattening).
    Arguments:
        - array: np.array
    Returns:
        - new_array: np.array
    """
    filter = ~np.isnan(array)
    new_array = np.vstack([row for row in [row_[filter[index]] for index, row_ in enumerate(array)] if len(row)>0])
    return new_array


#########################################
########### Specific functions ##########
#########################################

def get_subject_name(id):
    """ Get subject name from id.
    Arguments:
        - id: int
    Returns:
        - str
    """
    if id < 10:
        return 'sub-00{}'.format(id)
    elif id < 100:
        return 'sub-0{}'.format(id)
    else:
        return 'sub-{}'.format(id)

def filter_args(func, d):
    """ Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    Returns:
        - args: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys}
    return args

def output_name(folder_path, subject, model_name):
    """ Create a template name for the output deriving from
    given subject and model.
    Arguments:
        - folder_path: str
        - subject: str
        - model_name: str
    Returns:
        - template: str
    """
    folder = os.path.join(folder_path, subject, model_name)
    check_folder(folder)
    template = os.path.join(folder, '{}_{}_'.format(subject, model_name)
    return template

def possible_subjects_id(language):
    """ Returns possible subject id list for a given language.
    Arguments:
        - language: str
    Returns:
        result: list (of int)
    """
    if language=='english':
        result = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
                    94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    elif language=='french':
        result = [1] # TO DO
    elif language=='chineese':
        result = [1] # TO DO
    else:
        raise Exception('Language {} not known.'.format(language))
    
def fetch_offsets(offset_type, run_index):
    """ Retrieve the offset vector.
    Arguments:
        - offset_type: str
    Returns:
        - vector: np.array
    """
    offset_template_path = os.path.join(parameters['offset_path'], parameters['language'], 'onsets-offsets', '{offset_type}' + '_run{run_index}.csv')
    path = offset_template_path.format(offset_type=offset_type, run_index=run_index)
    if not os.path.exists:
        raise Exception("Please specify an offset file at: {}".format(path))
    else:
        offset = pd.read_csv(path, index=False).values
    return offset

def fetch_duration(duration_type, run_index, default_size=None):
    """ Retrieve the duration vector.
    Arguments:
        - duration_type: str
    Returns:
        - vector: np.array
    """
    duration_template_path = os.path.join(parameters['offset_path'], parameters['language'], 'durations', '{duration_type}' + '_run{run_index}.csv')
    path = duration_template_path.format(duration_type=duration_type, run_index=run_index)
    if not os.path.exists:
        duration = np.ones(default_size)
    else:
        duration = pd.read_csv(path, index=False).values
    return duration

def structuring_inputs(parameters):
    """ Structure inputs from submitted yaml file.
    Arguments:
        - parameters: dict
    Returns:
        - indexes: list (of np.array)
        - new_indexes: list (of np.array)
        - offset_type_dict: dict (of list)
        - duration_type_dict: dict (of list)
        - compression_types: list (of str)
        - n_components_list: list (of int)
    """
    indexes = []
    new_indexes = []
    nb_runs = parameters['nb_runs']
    offset_type_dict = {'run{}'.format(i): [] for i in range(1, nb_runs = 1)}
    duration_type_dict = {'run{}'.format(i): [] for i in range(1, nb_runs = 1)}
    compression_types = []
    n_components_list = []
    i = 0
    i_ = 0
    for index, model in enumerate(parameters['models']):
        compression_types.append(model['data_compression'] if model['data_compression'] else 'identity')
        n_components_list.append(model['ncomponents'])
        indexes.append(eval(model['columns_to_retrieve']) + i)
        i += len(eval(model['columns_to_retrieve']))
        offset_type_list.append(model['offset_type'])
        duration_type_list.append(model['duration_type'])
        if model['data_compression']:
            new_indexes.append(np.arange(i_, i_ + model['ncomponents']))
            i_ += model['ncomponents']
        else:
            new_indexes.append(eval(model['columns_to_retrieve']) + i_)
            i_ += len(eval(model['columns_to_retrieve']))
        for run_index in range(1, nb_runs + 1):
            offset_type_dict['run{}'.format(run_index)].append(model["offset_type"])
            duration_type_dict['run{}'.format(run_index)].append(duration_type=model["duration_type"])
    return indexes, new_indexes, offset_type_dict, duration_type_dict, compression_types, n_components_list
            

#########################################
########### Nilearn functions ###########
#########################################

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    """Returns a MultiNiftiMasker object from list (of list) of files.
    Arguments:
        - files: list (of list of str)
    Returns:
        - masker: MultiNiftiMasker
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
    masker.fit()
    return masker

def fetch_masker(masker_path, language):
    """ Fetch or compute if needed a global masker from all subjects of a
    given language.
    Arguments:
        - masker_path: str
        - language: str
    """
    if os.path.exists(masker_path):
        masker = nib.load(masker_path)
    else:
        fmri_runs = {}
        subjects = [get_subject_name(id) for id in possible_subjects_id(language)]
        for subject in subjects:
            fmri_path = os.path.join(paths.path2data, "fMRI", language, subject, "func")
            fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path, 'fMRI_*run*')))
        masker = compute_global_masker(list(fmri_runs.values()))
        nib.save(masker, masker_path)
    return masker
