import os
import yaml
import glob
import h5py
import json
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import nibabel as nib
from sklearn.linear_model import Ridge
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain, plot_img


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

def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
def write(path, text, end='\n'):
    """Write in the specified text file."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)

def save(object_to_save, path):
    """ Save an object to a given path.
    Arguments:
        - object_to_save: np.array / pd.DataFrame / dict (of np.array)
        - path: str
    """
    if isinstance(object_to_save, np.ndarray):
        extension = '.npy'
        np.save(path+extension, object_to_save)
    elif isinstance(object_to_save, pd.DataFrame):
        extension = '.csv'
        object_to_save.to_csv(path+extension, index=False)
    elif isinstance(object_to_save, dict):
        extension = '.hdf5'
        with h5py.File(path+extension, "w", libver='latest') as fout:
            for key in object_to_save.keys():
                if isinstance(object_to_save[key], np.ndarray):
                    fout.create_dataset(str(key), object_to_save[key].shape, data=object_to_save[key])
                elif isinstance(object_to_save[key], dict):
                    fout.create_dataset(str(key), data=json.dumps(object_to_save[key]))
                elif isinstance(object_to_save[key], list):
                    for index, arr in enumerate(object_to_save[key]):
                        if isinstance(arr, np.ndarray):
                            fout.create_dataset(str(key) + '_' + str(index), arr.shape, data=arr)
                        elif isinstance(arr, dict):
                            fout.create_dataset(str(key) + '_' + str(index), data=json.dumps(arr))
    elif isinstance(object_to_save, list):
        for index, item in enumerate(object_to_save):
            save(item, path + '_' + str(index))

def load(path):
    """ Load an object saved at a given path.
    Arguments:
        - path: str
    """
    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.csv'):
        data = pd.read_csv(path)
    elif path.endswith('.hdf5'):
        with h5py.File(path, "r", swmr=True) as fin:
            data = {key: json.loads(fin[key][()]) if isinstance(fin[key][()], str) else fin[key][()] for key in fin.keys()}
    elif path.endswith('.nii.gz'):
        data = nib.load(path)
    return data    

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
    filter_ = ~np.isnan(array)
    new_array = np.vstack([row for row in [row_[filter_[index]] for index, row_ in enumerate(array)] if len(row)>0])
    return new_array

def aggregate_cv(data):
    """ Transform a list of lists of dicts to
    a list of dicts of concatenated lists and
    aggregate accross splits
    Argument:
        - data: list (of list of dict)
    Returns:
        - result: list (of dict of list)
    """
    result = [{key: np.stack(np.array([dic[key] for dic in data[index]]), axis=0) for key in data[0][0]} for index in range(len(data))]
    return result


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

def get_nscans(language):
    result = {'english':{'run1':282,
                             'run2':298,
                             'run3':340,
                             'run4':303,
                             'run5':265,
                             'run6':343,
                             'run7':325,
                             'run8':292,
                             'run9':368
        },
                       'french':{'run1':309,
                             'run2':326,
                             'run3':354,
                             'run4':315,
                             'run5':293,
                             'run6':378,
                             'run7':332,
                             'run8':294,
                             'run9':336
        }
        }
    return result[language]

def filter_args(func, d):
    """ Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    Returns:
        - args: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys if key!='self'}
    return args

def get_output_name(folder_path, language, subject, model_name, data_name=''):
    """ Create a template name for the output deriving from
    given subject and model.
    Arguments:
        - folder_path: str
        - language: str
        - subject: str
        - model_name: str
        - data_name: str
    Returns:
        - output_path: str
    """
    folder = os.path.join(folder_path, language, subject, model_name)
    check_folder(folder)
    output_path = os.path.join(folder, '{}_{}_'.format(subject, model_name)) + data_name
    return output_path

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
    return result
    
def fetch_data(path_to_fmridata, path_to_input, subject, language, models=[]):
    """ Retrieve deep representations and fmri data.
    Arguments:
        - path_to_fmridata: str
        - path_to_input: str
        - subject: str
        - language: str
        - models: list (of dict)
    """
    fmri_path = os.path.join(path_to_fmridata, language, subject, "func")
    fMRI_paths = sorted(glob.glob(os.path.join(fmri_path, 'fMRI_*run*')))
    deep_representations_paths = [sorted(glob.glob(os.path.join(path_to_input, language, model['model_name'], model['input_template'] + '*run*.csv'))) for model in models]
    return deep_representations_paths, fMRI_paths

def fetch_offsets(offset_type, run_index, offset_path):
    """ Retrieve the offset vector.
    Arguments:
        - offset_type: str
        - run_index: int
        - offset_path: str
    Returns:
        - vector: np.array
    """
    index = run_index[-1]
    offset_template_path = os.path.join(offset_path, '{offset_type}' + '_run{index}.csv')
    path = offset_template_path.format(offset_type=offset_type, index=index)
    if not os.path.exists(path):
        raise Exception("Please specify an offset file at: {}".format(path))
    else:
        offset = pd.read_csv(path)['offsets'].values
    return offset

def fetch_duration(duration_type, run_index, duration_path, default_size=None):
    """ Retrieve the duration vector.
    Arguments:
        - duration_type: str
        - run_index: int
        - duration_path: str
        - default_size = int
    Returns:
        - vector: np.array
    """
    index = run_index[-1]
    duration_template_path = os.path.join(duration_path, 'durations', '{duration_type}' + '_run{index}.csv')
    path = duration_template_path.format(duration_type=duration_type, index=index)
    if not os.path.exists(path):
        duration = np.ones(default_size)
    else:
        duration = pd.read_csv(path).values
    return duration

def get_splitter_information(parameters):
    """ Retrieve the inputs for data transformation (make_regressor + standardization).
    Arguments:
        - parameters: dict
    Returns:
        - dict 
    """
    result = {'out_per_fold': parameters['nb_runs_test']}
    return result

def get_compression_information(parameters):
    """ Retrieve the inputs for data compression.
    Arguments:
        - parameters: dict
    Returns:
        - dict (indexes: list (of np.array),
        compression_types: list (of str),
        n_components_list: list (of int)
        )
    """
    indexes = []
    compression_types = []
    n_components_list = []
    i = 0
    for model in parameters['models']:
        compression_types.append(model['data_compression'] if model['data_compression'] else 'identity')
        n_components_list.append(model['ncomponents'])
        indexes.append(np.arange(i, len(eval(model['columns_to_retrieve'])) + i))
        i += len(eval(model['columns_to_retrieve']))
    return {'indexes': indexes, 'compression_types': compression_types, 'n_components_list': n_components_list}

def get_data_transformation_information(parameters):
    """ Retrieve the inputs for data transformation (make_regressor + standardization).
    Arguments:
        - parameters: dict
    Returns:
        - dict (new_indexes: list (of np.array),
        offset_type_dict: dict (of list),
        duration_type_dict: dict (of list)
        )
    """
    nb_runs = parameters['nb_runs']
    new_indexes = []
    offset_type_dict = {'run{}'.format(i): [] for i in range(1, nb_runs + 1)}
    duration_type_dict = {'run{}'.format(i): [] for i in range(1, nb_runs + 1)}
    i_ = 0
    for model in parameters['models']:
        if model['data_compression']:
            new_indexes.append(np.arange(i_, i_ + model['ncomponents']))
            i_ += model['ncomponents']
        else:
            new_indexes.append(np.arange(i_, len(eval(model['columns_to_retrieve'])) + i_))
            i_ += len(eval(model['columns_to_retrieve']))
        for run_index in range(1, nb_runs + 1):
            offset_type_dict['run{}'.format(run_index)].append(model["offset_type"])
            duration_type_dict['run{}'.format(run_index)].append(model["duration_type"])
    result =  {'indexes': new_indexes, 'offset_type_dict': offset_type_dict, 'duration_type_dict': duration_type_dict,
                'tr': parameters['tr'], 'nscans': get_nscans(parameters['language']), 
                'offset_path': parameters['offset_path'], 'duration_path': parameters['duration_path'], 
                'language': parameters['language'], 'hrf': parameters['hrf']}
    return result
            
def get_encoding_model_information(parameters):
    """ Retrieve the inputs for data transformation (make_regressor + standardization).
    Arguments:
        - parameters: dict
    Returns:
        - dict
    """
    result = {'model': eval(parameters['encoding_model']), 'alpha': parameters['alpha'], 
                'alpha_min_log_scale': parameters['alpha_min_log_scale'], 
                'alpha_max_log_scale': parameters['alpha_max_log_scale'], 
                'nb_alphas': parameters['nb_alphas'], 'optimizing_criteria': parameters['optimizing_criteria'],
                'base': parameters['base']}
    return result

#########################################
########### Nilearn functions ###########
#########################################

def compute_global_masker(files, smoothing_fwhm=None): # [[path, path2], [path3, path4]]
    """Returns a MultiNiftiMasker object from list (of list) of files.
    Arguments:
        - files: list (of list of str)
    Returns:
        - masker: MultiNiftiMasker
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, smoothing_fwhm=smoothing_fwhm)
    masker.fit()
    return masker

def fetch_masker(masker_path, language, path_to_fmridata, path_to_input, smoothing_fwhm=None, logger=None):
    """ Fetch or compute if needed a global masker from all subjects of a
    given language.
    Arguments:
        - masker_path: str
        - language: str
        - path_to_input: str
        - path_to_fmridata: str
        - smoothing_fwhm: int
        - logger: Logger
    """
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        logger.report_state(" loading existing masker...")
        params = read_yaml(masker_path + '.yml')
        mask_img = nib.load(masker_path + '.nii.gz')
        masker = MultiNiftiMasker()
        masker.set_params(**params)
        masker.fit([mask_img])
    else:
        logger.report_state(" recomputing masker...")
        fmri_runs = {}
        subjects = [get_subject_name(id) for id in possible_subjects_id(language)]
        for subject in subjects:
            _, fmri_paths = fetch_data(path_to_fmridata, path_to_input, subject, language)
            fmri_runs[subject] = fmri_paths
        masker = compute_global_masker(list(fmri_runs.values()), smoothing_fwhm=smoothing_fwhm)
        params = masker.get_params()
        params = {key: params[key] for key in ['detrend', 'dtype', 'high_pass', 'low_pass', 'mask_strategy', 
                                                'memory_level', 'n_jobs', 'smoothing_fwhm', 'standardize',
                                                't_r', 'verbose']}
        nib.save(masker.mask_img_, masker_path + '.nii.gz')
        save_yaml(params, masker_path + '.yml')
    return masker

def create_maps(masker, distribution, output_path, vmax=None, not_glass_brain=False, logger=None, distribution_max=None, distribution_min=None):
    """ Create the maps from the distribution.
    Arguments:
        - masker: NifitMasker
        - distribution: np.array (1D)
        - output_path: str
        - vmax: float
        - not_glass_brain: bool
    """
    logger.info("Transforming array to .nii image...")
    if distribution_min is not None:
        if distribution_max is not None:
            mask = np.where((distribution < distribution_min) | (distribution > distribution_max))
        else:
            mask = np.where(distribution < distribution_min)
        distribution[mask] = np.nan # remove outliers
    else:
        mask = np.where(distribution > distribution_max)
        distribution[mask] = np.nan # remove outliers
    img = masker.inverse_transform(distribution)
    logger.validate()
    logger.info("Saving image...")
    nib.save(img, output_path + '.nii.gz')
    logger.validate()

    plt.hist(distribution[~np.isnan(distribution)], bins=50)
    plt.savefig(output_path + '_hist.png')
    plt.close()

    logger.info("Saving glass brain...")
    if not_glass_brain:
        display = plot_img(img, colorbar=True, black_bg=True, cut_coords=(-48, 24, -10))
        display.savefig(output_path + '.png')
        display.close()
    else:
        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=True, vmax=vmax, plot_abs=False)
        display.savefig(output_path + '.png')
        display.close()
    logger.validate()
