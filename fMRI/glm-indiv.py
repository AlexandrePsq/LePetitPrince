############################################
# First level analysis 
# --> R2 maps for GLM models
#
############################################


import argparse
from os.path import join
import os.path as op

import warnings
warnings.simplefilter(action='ignore' )

from ..utilities.settings import Paths, Subjects
from ..utilities.utils import *
import pandas as pd
from nilearn.masking import compute_epi_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from nilearn.image import math_img, mean_img
from joblib import Parallel, delayed

paths = Paths()
subjects_list = Subjects()


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0)
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def compute_crossvalidated_r2(fmri_runs, design_matrices, subject):

    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None

    logo = LeaveOneGroupOut() # leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train]) # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train = np.vstack([design_matrices[i] for i in train])
        model = LinearRegression().fit(predictors_train, fmri_data_train)

        # return the R2_score for each voxel (=list)
        r2_train_ = get_r2_score(model, fmri_data_train, predictors_train)
        r2_test_ = get_r2_score(model, fmri_runs[test[0]], design_matrices[test[0]])

        # log the results
        log(subject, 'train', r2_train_)
        log(subject, 'test', r2_test_)

        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, r2_train_])
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, r2_test_])

    return (np.mean(r2_train, axis=0), np.mean(r2_test, axis=0)) # compute mean vertically (in time)


def do_single_subject(subject, fmri_runs, matrices, masker, output_parent_folder):
    # Compute r2 maps for all subject for a given model
    #   - subject : e.g. : 'sub-060'
    #   - fmri_runs: list of fMRI data runs (1 for each run)
    #   - matrices: list of design matrices (1 for each run)
    #   - masker: MultiNiftiMasker object

    fmri_runs = [masker.transform(f) for f in fmri_filenames] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column

    # compute r2 maps and save them under .nii.gz and .png formats
    r2_train, r2_test = compute_crossvalidated_r2(fmri_runs, matrices, subject)
    create_r2_maps(masker, r2_train, 'train', subject, output_parent_folder) # r2 train
    create_r2_maps(masker, r2_test, 'test', subject, output_parent_folder) # r2 test
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", type=bool, default=True, help="Precise if we run the code in parallel")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'glm-indiv'

    for model_ in args.models[0]:
        subjects = subjects_list.get_all(args.language, args.test)
        dm = get_data(args.language, input_data_type, model=model_, source='fMRI', test=args.test)
        fmri_runs = {subject: get_data(args.language, data_type=source, test=args.test, subject=subject) for subject in subjects}

        output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

        matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1
        masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object ... computation is sloow

        if args.parallel:
                Parallel(n_jobs=-1)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, output_parent_folder) for sub in subjects)
            else:
                for sub in subjects:
                    print(f'Processing subject {}...'.format(sub))
                    do_single_subject(sub, fmri_runs[sub], matrices, masker, output_parent_folder)
