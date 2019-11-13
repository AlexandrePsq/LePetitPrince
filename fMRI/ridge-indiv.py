############################################
# First level analysis 
# --> R2 and Pearson maps for GLM models
#
############################################

import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
from os.path import join
import os.path as op

import warnings
warnings.simplefilter(action='ignore')

from utilities.settings import Paths, Subjects, Params
from utilities.utils import get_data, transform_design_matrices, pca
from utilities.first_level_analysis import compute_global_masker, do_single_subject
import pandas as pd
from nilearn.masking import compute_epi_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge, RidgeCV
from nilearn.image import math_img, mean_img
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.switch_backend('agg')

paths = Paths()
params = Params()
subjects_list = Subjects()



if __name__ == '__main__':
    """First level analysis using a Ridge model.
    We fit the design-matrices to the fMRI data for each subject and cross-validate 
    accross the runs. PCA transform can be applied, but we recommend not to as 
    interpretability would be damaged and the intrinsic regulation of the Ridge model 
    will do the work.
    """

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--subjects", nargs='+', action='append', default=[], help="Subjects list on whom we are running a test: list of 'sub-002...")
    parser.add_argument("--language", type=str, default='english', help="Language of the model.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use to generate the raw features.")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we run the code in parallel")
    parser.add_argument("--voxel_wised", default=False, action='store_true', help="Precise if we compute voxel-wised")
    parser.add_argument("--alphas", nargs='+', action='append', default=[np.logspace(-3, 3, 30)], help="List of alphas for voxel-wised analysis.")
    parser.add_argument("--pca", type=int, default=params.n_components_default, help="pca value to use.")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'ridge-indiv'
    alphas = args.alphas[0]
    model = Ridge() if args.voxel_wised else Ridge(params.pref.alpha_default) #RidgeCV(params.pref.alpha_default, scoring='r2')
    model_name = args.model_name
    subjects = args.subjects[0]


    # Retrieving data
    dm = get_data(args.language, input_data_type, model=model_name, source='fMRI')
    fmri_runs = {subject: get_data(args.language, data_type=source, subject=subject) for subject in subjects}

    matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1

    # PCA transformation
    if (matrices[0].shape[1] > args.pca) & (params.pca):
        print('PCA analysis running...')
        matrices = pca(matrices, model_name, n_components=args.pca)
        print('PCA done.')
    else:
        print('Skipping PCA.')

    # Computing masker
    masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object ... computation is sloow


    # Fitting the Ridge model
    if args.parallel:
            Parallel(n_jobs=-2)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, source, model, voxel_wised=args.voxel_wised, alpha_list=alphas, pca=args.pca) for sub in subjects)
    else:
        for sub in subjects:
            print('Processing subject {}...'.format(sub))
            do_single_subject(sub, fmri_runs[sub], matrices, masker, source, model, voxel_wised=args.voxel_wised, alpha_list=alphas, pca=args.pca)
