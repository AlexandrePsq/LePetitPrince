import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
from os.path import join


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utilities.settings import Paths, Scans
from utilities.utils import *
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nistats.hemodynamic_models import compute_regressor


paths = Paths()
scans = Scans()


def process_raw_features(run, tr, nscans):
    # compute convolution of raw_features with hrf kernel
    df = pd.read_csv(run)
    conditions = np.vstack((df.onsets, df.duration, df.amplitude))
    result = compute_regressor(exp_condition=conditions,
                               hrf_model='spm',
                               frame_times=np.arange(0.0, nscans*tr, tr),
                               oversampling=10)
    return pd.DataFrame(result[0], columns=['hrf'])

def compute_features(run, output_parent_folder, output_data_type, language, model, extension, tr, overwrite):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    nscans = scans.get_nscans(language, run_name) # retrieve the number of scans for the run
    path2output = get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        df = process_raw_features(run, tr, nscans)
        df.to_csv(path2output, index=False, header=False) # saving features.csv


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate features (=fMRI regressors) from raw_features (csv file with 3 columns onset-amplitude-duration) by convolution with an hrf kernel.\n\nInput:\nTime resolution of the scans, language and models.""")
    parser.add_argument("--tr", type=float, default=None, help="TR in sec (=sampling period for scans)")
    parser.add_argument("--model", type=str, help="Names of the model used to generate the raw features")
    parser.add_argument("--language", type=str, default='en', help="Language of the model studied.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we want to run code in parallel")

    args = parser.parse_args()

    input_data_type = 'raw_features'
    output_data_type = 'features'
    extension = '.csv'
    source = 'fMRI'
    model = args.model

    output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model)
    check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

    raw_features = get_data(args.language, input_data_type, model=model, source='fMRI')
    
    if not args.parallel:
        for i, run in enumerate(raw_features):
            compute_features(run, output_parent_folder, output_data_type, args.language, model, extension, args.tr, args.overwrite)
    else:
        Parallel(n_jobs=-2)(delayed(compute_features) \
                    (run, output_parent_folder, output_data_type, args.language, model, extension, args.tr, args.overwrite) for run in raw_features)


