############################################
# First level analysis 
# --> R2 maps for GLM models
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
from utilities.utils import get_data, get_output_parent_folder, check_folder, transform_design_matrices, pca
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

paths = Paths()
params = Params()
subjects_list = Subjects()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--subjects", nargs='+', action='append', default=[], help="Subjects list on whom we are running a test: list of 'sub-002...")
    parser.add_argument("--language", type=str, default='english', help="Language of the model.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=True, action='store_true', help="Precise if we run the code in parallel")
    parser.add_argument("--voxel_wised", default=False, action='store_true', help="Precise if we compute voxel-wised")
    parser.add_argument("--alphas", nargs='+', action='append', default=[np.logspace(-3, 1, 30)], help="List of alphas for voxel-wised analysis.")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'ridge-indiv'
    alphas = args.alphas[0]
    model = Ridge() if args.voxel_wised else RidgeCV(alphas, scoring='r2')
    model_name = args.model_name
    subjects = args.subjects[0]

    dm = get_data(args.language, input_data_type, model=model_name, source='fMRI')
    fmri_runs = {subject: get_data(args.language, data_type=source, subject=subject) for subject in subjects}

    output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_name)
    check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

    matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1
    print('PCA analysis running...')
    matrices = pca(matrices, n_components=params.n_components)
    print('PCA done.')
    masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object ... computation is sloow

    if args.parallel:
            Parallel(n_jobs=-1)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, output_parent_folder, model, voxel_wised=args.voxel_wised, alpha_list=alphas) for sub in subjects)
    else:
        for sub in subjects:
            print('Processing subject {}...'.format(sub))
            do_single_subject(sub, fmri_runs[sub], matrices, masker, output_parent_folder, model, voxel_wised=args.voxel_wised, alpha_list=alphas)
