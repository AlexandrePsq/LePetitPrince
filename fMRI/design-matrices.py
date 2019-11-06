########
# Merge features columns into a matrix.
#
########

import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
from os.path import join
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore' )

from utilities.settings import Paths, Params
from utilities.utils import get_data, check_folder, compute, get_output_parent_folder, get_path2output
import pandas as pd

paths = Paths()
params = Params()


if __name__ == '__main__':
    """Create the design matrix for each run for a given aggregated model (ensemble of models).
    An aggregated model associated with a design-matrix can contain various smaller models.
    The names should be separated by a '+' (without spaces, e.g.: 'rms_model+wordrate_model')
    To create the design-matrix, we concatenate the features dataframes of the constituants of the 
    aggregated model and scale their mean and variance (by column).
    """

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--language", type=str, default='english', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")

    args = parser.parse_args()

    features_list = []
    models = args.models[0]
    last_step = 'features'
    step = 'design-matrices'
    extension = '.csv'
    source = 'fMRI'

    # Retrieving data
    for model_ in models:
        features_list.append(get_data(args.language, last_step, model=model_, source='fMRI')) # retrieve the data to transform and append the list of runs (features data) to features_list) 

    runs = list(zip(*features_list)) # list of 9 tuples (1 for each run), each tuple containing the features for all the specified models
    # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]


    # Computing design-matrices
    for i in range(len(runs)):
        model_name = '+'.join(models)
        output_parent_folder = get_output_parent_folder(source, step, args.language, model_name)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not
        name = os.path.basename(os.path.splitext(runs[i][0])[0])
        run_name = name.split('_')[-1] # extract the name of the run
        path2output = get_path2output(output_parent_folder, step, args.language, model_name, run_name, extension)

        if compute(path2output, overwrite=args.overwrite):
            merge = pd.concat([pd.read_csv(path2features, header=0) for path2features in runs[i]], axis=1) # concatenate horizontaly the read csv files of a run
            matrices = merge.values
            scaler = StandardScaler(with_mean=params.scaling_mean, with_std=params.scaling_var)
            scaler.fit(matrices)
            matrices = scaler.transform(matrices)
            result = pd.DataFrame(matrices, columns=merge.columns)
            result.to_csv(path2output, index=False)
