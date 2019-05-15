import glob
import os
from os.path import join
from .settings import Paths, Extensions

paths = Paths()
extensions = Extensions()


def get_data(language, data_type, source='', model='', test=False, subject):
    # General function for data retrieving
    # Output: list of path to the different data files
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    if data_type in sub_dir:
        base_path = paths.path2data
        if data_type in ['fMRI', 'MEG']:
            file_pattern = '{2}/func/{0}_{1}_{2}'.format(data_type, language, subject) + '_run*' + extension
        else:
            file_pattern = '{}_{}'.format(data_type, language) + '_run*' + extension
    else:
        base_path = join(paths.path2derivatives, source)
        file_pattern = '{}_{}'.format(data_type, language) + '_' + model + '_run*' + extension
    if test:
        data = [join(base_path, '{0}/{1}/test/{2}'.format(data_type, language, data_type)+extension)]
    else:
        data = [sorted(glob.glob(join(base_path, '{0}/{1}/{2}'.format(data_type, language, model), file_pattern)))]
    return data


def compute(path, overwrite=False):
    # Tell us if we can compute or not
    result = True
    if os.path.isfile(path):
        result = overwrite
    return result

def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        os.mkdir(path)