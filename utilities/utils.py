import glob
import os
from os.path import join
from .settings import Paths, Extensions


from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

paths = Paths()
extensions = Extensions()


#########################################
############ Data retrieving ############
#########################################

def get_data(language, data_type, subject=None, source='', model=''):
    # General function for data retrieving
    # Output: list of path to the different data files
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    if data_type in sub_dir:
        base_path = paths.path2data
        if data_type in ['fMRI', 'MEG']:
            file_pattern = '{2}/func/{0}_{1}_{2}_run*'.format(data_type, language, subject) + extension
        else:
            file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    else:
        base_path = join(paths.path2derivatives, source)
        file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    data = sorted(glob.glob(join(base_path, '{0}/{1}/{2}'.format(data_type, language, model), file_pattern)))
    return data


def get_output_parent_folder(source, output_data_type, language, model):
    return join(paths.path2derivatives, '{0}/{1}/{2}/{3}'.format(source, output_data_type, language, model))


def get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension):
    return join(output_parent_folder, '{0}_{1}_{2}_{3}'.format(output_data_type, language, model, run_name) + extension)



#########################################
###### Computation functionalities ######
#########################################

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



#########################################
################## Log ##################
#########################################

def log(subject, voxel, alpha, r2):
    """ log stats per fold to a csv file """
    logcsvwriter = csv.writer(open("test.log", "a+"))
    if voxel == 'whole brain':
        logcsvwriter.writerow([subject, voxel, np.mean(r2), np.std(r2),
                            np.min(r2), np.max(r2)])
    else:
        logcsvwriter.writerow([subject, voxel, alpha, r2])


#########################################
########## Classical functions ##########
#########################################


def get_r2_score(model, y_true, y2predict, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    r2 = r2_score(y_true,
                    model.predict(y2predict),
                    multioutput='raw_values')
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0).values
    scaler = StandardScaler()
    scaler.fit(dm)
    dm = scaler.transform(dm)
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 


