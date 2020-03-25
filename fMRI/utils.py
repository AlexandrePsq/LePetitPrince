import os
import yaml
import inspect
import numpy as np
import pandas as pd

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


#########################################
########### Specific functions ##########
#########################################

def get_subject_name(id):
    """ Get subject name from id."""
    if id < 10:
        return 'sub-00{}'.format(id)
    elif id < 100:
        return 'sub-0{}'.format(id)
    else:
        return 'sub-{}'.format(id)

def merge_dict(list_of_dict):
    """ Merge a list of dictionaries into a single dictionary.
    Arguments:
        - list_of_dict: list (of dicts)
    """
    result = {key: value for d in list_of_dict for key, value in d.items()}
    return result

def filter_args(func, d):
    """ Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys}
    return args

def output_name(folder_path, subject, model_name):
    folder = os.path.join(folder_path, subject, model_name)
    check_folder(folder)
    template = os.path.join(folder, '{}_{}_'.format(subject, model_name)
    return template
    

#########################################
########### Nilearn functions ###########
#########################################

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    """Returns a MultiNiftiMasker object from list (of list) of files."""
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
    masker.fit()
    return masker
