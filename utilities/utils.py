import glob
import os
from os.path import join
from .settings import Paths, Extensions

paths = Paths()
extensions = Extensions()


def get_data(language, data_type, source='', model=None, test=False):
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    if data_type in sub_dir:
        base_path = paths.path2data
        file_pattern = '{}_{}'.format(data_type, language) + '_run*' + extension
    else:
        base_path = join(paths.path2derivatives, source)
        file_pattern = '{}_{}'.format(data_type, language) + '_' + model + '_run*' + extension
    if test:
        data = [join(base_path, '{0}/{1}/test/{2}.csv'.format(data_type, language, data_type))]
    else:
        data = [sorted(glob.glob(join(base_path, '{0}/{1}/{2}'.format(data_type, language, model), file_pattern)))]
    return data
