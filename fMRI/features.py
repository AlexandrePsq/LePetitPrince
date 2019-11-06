import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
from os.path import join


import warnings
warnings.simplefilter(action='ignore')

from utilities.settings import Paths, Scans
from utilities.utils import get_data, check_folder, compute, get_output_parent_folder, get_path2output
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nistats.hemodynamic_models import compute_regressor


paths = Paths()
scans = Scans()


def process_raw_features(run, tr, nscans):
    """For each column in the raw-features, compute the convolution
    with an hrf kernel specified in settings.py
    :run: (str) path to the raw-feature file
    :tr: (float) sampling period for scan
    :nscans: (int) number of scans to generate
    """
    # compute convolution of raw_features with hrf kernel
    df = pd.read_csv(run)
    result = []
    raw_features_columns = [col for col in df.columns if col not in ['offsets', 'duration']]
    for col in raw_features_columns:
        conditions = np.vstack((df.offsets, df.duration, df[col]))
        tmp = compute_regressor(exp_condition=conditions,
                                hrf_model='spm',
                                frame_times=np.arange(0.0, nscans*tr, tr),
                                oversampling=10)
        result.append(pd.DataFrame(tmp[0], columns=[col]))
    return pd.concat(result, axis=1)

def compute_features(run, output_parent_folder, step, language, model_name, extension, tr, overwrite):
    """Compute features (regressors) from raw-features.
    :run: (str) path to the raw-feature file
    :tr: (float) sampling period for scan
    :output_parent_folder: (str) path to the folder where derivatives will be saved
    :step: (str) step of the pipeline that is being computed
    :language: (str) language 
    :model_name: (str) name of the model being studied
    :extension: (str) extension used for the derivatives that will be saved
    :overwrite: (boolean) for overwriting existing data
    """
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    nscans = scans.get_nscans(language, run_name) # retrieve the number of scans for the run
    path2output = get_path2output(output_parent_folder, step, language, model_name, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        df = process_raw_features(run, tr, nscans)
        df.to_csv(path2output, index=False) # saving features.csv


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate features (=fMRI regressors) from raw_features (csv file with 3 columns onset-amplitude-duration) by convolution with an hrf kernel.\n\nInput:\nTime resolution of the scans, language and models.""")
    parser.add_argument("--tr", type=float, default=None, help="TR in sec (=sampling period for scans)")
    parser.add_argument("--model", type=str, help="Names of the model used to generate the raw features")
    parser.add_argument("--language", type=str, default='english', help="Language of the model studied.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we want to run code in parallel")

    args = parser.parse_args()

    last_step = 'raw-features'
    step = 'features'
    extension = '.csv'
    source = 'fMRI'
    model = args.model


    # Retrieving data
    output_parent_folder = get_output_parent_folder(source, step, args.language, model)
    check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

    raw_features = get_data(args.language, last_step, model=model, source='fMRI')
    

    # Computing Features
    if not args.parallel:
        for i, run in enumerate(raw_features):
            compute_features(run, output_parent_folder, step, args.language, model, extension, args.tr, args.overwrite)
    else:
        Parallel(n_jobs=-2)(delayed(compute_features) \
                    (run, output_parent_folder, step, args.language, model, extension, args.tr, args.overwrite) for run in raw_features)


