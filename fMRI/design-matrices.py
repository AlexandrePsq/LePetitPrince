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
from utilities.utils import get_data, compute, get_path2output, standardization
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
    
    model_name = '+'.join(models)
    dataframes = []
    matrices = []
    evaluate = True

    # Computing design-matrices
    for i in range(len(runs)):
        name = os.path.basename(os.path.splitext(runs[i][0])[0])
        run_name = name.split('_')[-1] # extract the name of the run
        path2output = get_path2output(source, step, args.language, model_name, run_name, extension)
        evaluate = (evaluate and compute(path2output, overwrite=args.overwrite))

        merge = pd.concat([pd.read_csv(path2features, header=0) for path2features in runs[i]], axis=1) # concatenate horizontaly the read csv files of a run
        dm = merge.values
        dataframes.append(pd.DataFrame(dm, columns=merge.columns))
        matrices.append(dm)
    
    matrices = standardization(matrices, model_name, pca_components=300) if evaluate else matrices

    for i in range(len(runs)):
        name = os.path.basename(os.path.splitext(runs[i][0])[0])
        run_name = name.split('_')[-1] # extract the name of the run
        path2output = get_path2output(source, step, args.language, model_name, run_name, extension)

        if compute(path2output, overwrite=args.overwrite):
            result = pd.DataFrame(matrices[i], columns=['{}_component-{}'.format(model_name, index) for index in range(params.n_components_default)])
            result.to_csv(path2output, index=False)
