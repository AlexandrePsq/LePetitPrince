import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
from os.path import join


import warnings
warnings.simplefilter(action='ignore')

from utilities.settings import Paths, Scans, Params
from utilities.utils import get_data, compute, get_path2output, standardization, scale, get_category
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nistats.hemodynamic_models import compute_regressor


paths = Paths()
scans = Scans()
params = Params()


def preprocess_predictions(runs, model_name, model_category, onsets_paths, language, pca=False, pca_type='simple', pca_components=300, extension='.csv', shift_surprisal=None):
    """Apply PCA to the predictions if requested and add offset and 
    duration columns.
    :runs: (list) list of paths to the prediction of each run.
    :language: (str) language used.
    :pca: (bool) Precise if we apply pca.
    :pca_type: (str) Type of PCA used.
    :pca_components: (int) Number of components to keep in the PCA.
    :model_name: (str) name of the model being studied.
    :model_category: (str) category of the model being studied.
    :extension: (str) extension used for the derivatives that will be saved.
    """
    data = [pd.read_csv(run) for run in runs]
    if pca:
        matrices = [df.values for df in data]
        matrices = standardization(matrices, model_name, pca_components=300, scaling=False)
        data = [pd.DataFrame(matrix, columns=['{}_component-{}'.format(model_name, index) for index in range(pca_components)]) for matrix in matrices]
    for run_index, run in enumerate(runs):
        textgrid = pd.read_csv(onsets_paths[run_index]) # df with onsets-offsets-word 
        data[run_index]['offsets'] = textgrid['offsets']
        data[run_index]['duration'] = 0
        if shift_surprisal:
            data[run_index]['offsets'] += shift_surprisal
    return data

def process_predictions(array, tr, nscans):
    """For each column in the raw-features, compute the convolution
    with an hrf kernel specified in settings.py.
    :array: (dataframe) preprocessed predictions of the model with onset.
    :tr: (float) sampling period for scan.
    :nscans: (int) number of scans to generate.
    """
    # compute convolution of raw_features with hrf kernel
    result = []
    raw_features_columns = [col for col in array.columns if col not in ['offsets', 'duration']]
    for col in raw_features_columns:
        conditions = np.vstack((array.offsets, array.duration, array[col]))
        tmp = compute_regressor(exp_condition=conditions,
                                hrf_model='spm',
                                frame_times=np.arange(0.0, nscans*tr, tr),
                                oversampling=10)
        result.append(pd.DataFrame(tmp[0], columns=[col]))
    return pd.concat(result, axis=1)

def compute_features(run, index, source, step, language, model_name, extension, tr, overwrite):
    """Compute features (regressors) from predictions.
    :run: (dataframe) preprocessed predictions of the model with onset.
    :index: (int) index of the run.
    :tr: (float) sampling period for scan.
    :source: (str) Source of acquisition used (fMRI or MEG).
    :step: (str) step of the pipeline that is being computed.
    :language: (str) language used.
    :model_name: (str) name of the model being studied.
    :extension: (str) extension used for the derivatives that will be saved.
    :overwrite: (boolean) for overwriting existing data.
    """
    run_name = 'run{}'.format(index) # extract the name of the run
    nscans = scans.get_nscans(language, run_name) # retrieve the number of scans for the run
    path2output = get_path2output(source, step, language, model_name, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        df = process_predictions(run, tr, nscans)
        return df, path2output


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate features (=fMRI regressors) from raw_features (csv file with 3 columns onset-amplitude-duration) by convolution with an hrf kernel.\n\nInput:\nTime resolution of the scans, language and models.""")
    parser.add_argument("--tr", type=float, default=None, help="TR in sec (=sampling period for scans)")
    parser.add_argument("--model", type=str, help="Names of the model used to generate the raw features")
    parser.add_argument("--language", type=str, default='english', help="Language of the model studied.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we want to run code in parallel")
    parser.add_argument("--shift_surprisal", default=None, type=int, help="Precise how we shape surprisal.")
    parser.add_argument("--onsets_paths",  nargs='+', action='append', help="Path to the specific onsets file.")

    args = parser.parse_args()

    last_step = 'predictions'
    step = 'features'
    extension = '.csv'
    source = 'fMRI'
    model = args.model
    model_category = get_category(model)
    onsets_paths = args.onsets_paths[0]


    # Retrieving data
    predictions = get_data(args.language, last_step, model=model, source='fMRI')

    # Preprocessing the predictions (PCA + add onsets & duration)
    raw_features = preprocess_predictions(predictions, model, model_category, onsets_paths, args.language, pca=False, extension=extension, shift_surprisal=args.shift_surprisal)

    # Computing Features
    if not args.parallel:
        result = [compute_features(array, index+1, source, step, args.language, model, extension, args.tr, args.overwrite) for index, array in enumerate(raw_features)]
    else:
        result = Parallel(n_jobs=-2)(delayed(compute_features) \
                        (array, index+1, source, step, args.language, model, extension, args.tr, args.overwrite) for index, array in enumerate(raw_features))
    
    # Scaling
    data, paths = list(zip(*result))
    matrices = scale([array.values for array in data])
    data = [pd.DataFrame(data=matrix, columns=data[index].columns) for index, matrix in enumerate(matrices)]

    # Saving
    for index, df in enumerate(data):
        df.to_csv(paths[index], index=False)
