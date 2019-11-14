############################################
# First level analysis 
# --> R2 and pearson maps for GLM models
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
warnings.simplefilter(action='ignore' )

from utilities.settings import Paths, Subjects, Params
from utilities.utils import get_data, transform_design_matrices, pca
from utilities.first_level_analysis import compute_global_masker, do_single_subject
import pandas as pd
import numpy as np
from nilearn.input_data import MultiNiftiMasker
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.switch_backend('agg')

paths = Paths()
params = Params()
subjects_list = Subjects()


if __name__ == '__main__':
    """First level analysis using a GLM.
    We fit the design-matrices to the fMRI data for each subject and cross-validate 
    accross the runs. PCA transform can be applied, but we recommend not to as 
    interpretability would be damaged (use Ridge model instead).
    """

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--subjects", nargs='+', action='append', default=[], help="Subjects list on whom we are running a test: list of 'sub-002...")
    parser.add_argument("--language", type=str, default='english', help="Language of the model.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we run the code in parallel")
    parser.add_argument("--pca", type=int, default=params.n_components_default, help="pca value to use.")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'glm-indiv'
    model = LinearRegression()
    model_name = args.model_name
    subjects = args.subjects[0]


    # Retrieving data
    dm = get_data(args.language, input_data_type, model=model_name, source='fMRI')
    if (not params.force_glm) and (len(dm)==0 or len(pd.read_csv(dm[0]).columns) > 20):
        print("You have a lot of features: Linear regression will give you an empty mask (highly negative r2)... you should use a Ridge regression")
        quit()
    fmri_runs = {subject: get_data(args.language, data_type=source, subject=subject) for subject in subjects}

    matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1

    # Computing masker
    masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object 

    # Fitting the GLM model
    if args.parallel:
            Parallel(n_jobs=-2)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, source, input_data_type, args.language, model, pca=args.pca) for sub in subjects)
    else:
        for sub in subjects:
            print('Processing subject {}...'.format(sub))
            do_single_subject(sub, fmri_runs[sub], matrices, masker, source, input_data_type, args.language, model, pca=args.pca)
