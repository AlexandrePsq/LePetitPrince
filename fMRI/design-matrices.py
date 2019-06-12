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

import warnings
warnings.simplefilter(action='ignore' )

from utilities.settings import Paths
from utilities.utils import get_data, check_folder, compute, get_output_parent_folder, get_path2output
import pandas as pd

paths = Paths()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--language", type=str, default='english', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")

    args = parser.parse_args()

    features_list = []
    models = sorted(args.models[0])
    input_data_type = 'features'
    output_data_type = 'design-matrices'
    extension = '.csv'
    source = 'fMRI'

    for model_ in models:
        features_list.append(get_data(args.language, input_data_type, model=model_, source='fMRI')) # retrieve the data to transform and append the list of runs (features data) to features_list) 

    runs = list(zip(*features_list)) # list of 9 tuples (1 for each run), each tuple containing the features for all the specified models
    # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]

    for i in range(len(runs)):
        # create the design matrix for each run
        model_name = '-'.join(models)
        output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_name)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not
        name = os.path.basename(os.path.splitext(runs[i][0])[0])
        run_name = name.split('_')[-1] # extract the name of the run
        path2output = get_path2output(output_parent_folder, output_data_type, args.language, model_name, run_name, extension)

        if compute(path2output, overwrite=args.overwrite):
            merge = pd.concat([pd.read_csv(path2features, header=0) for path2features in runs[i]], axis=1) # concatenate horizontaly the read csv files of a run
            merge.to_csv(path2output, index=False)
