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
from sklearn.linear_model import LinearRegression
from nilearn.image import math_img, mean_img
from joblib import Parallel, delayed

paths = Paths()
subjects_list = Subjects()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", type=bool, default=True, help="Precise if we run the code in parallel")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'glm-indiv'
    model = LinearRegression()

    for model_ in args.models[0]:
        subjects = subjects_list.get_all(args.language, args.test)
        dm = get_data(args.language, input_data_type, model=model_, source='fMRI', test=args.test)
        fmri_runs = {subject: get_data(args.language, data_type=source, test=args.test, subject=subject) for subject in subjects}

        output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

        matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1
        masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object ... computation is sloow

        if args.parallel:
                Parallel(n_jobs=-1)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, output_parent_folder, model) for sub in subjects)
            else:
                for sub in subjects:
                    print(f'Processing subject {}...'.format(sub))
                    do_single_subject(sub, fmri_runs[sub], matrices, masker, output_parent_folder, model)
