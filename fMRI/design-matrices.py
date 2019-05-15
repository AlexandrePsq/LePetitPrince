########
# Merge features columns into a matrix.
#
########

import argparse
from os.path import join

import warnings
warnings.simplefilter(action='ignore' )

from ..utilities.settings import Paths
from ..utilities.utils import *
import pandas as pd

paths = Paths()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")

    args = parser.parse_args()

    data_list = []
    models = sorted(args.models[0])

    for model_ in models:
        data_type = 'features'
        extension = '.csv'
        data_list.append(get_data(args.language, data_type, model_, source='fMRI', args.test)) # retrieve the data to transform and append the list of runs (features data) to data_list) 

    runs = list(zip(*data_list)) # list of 9 tuples (1 for each run), each tuple containing the features for all the specified models
    # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]

    for i in range(len(runs)):
        # create the design matrix for each run
        model_name = '_'.join(models)
        output_parent_folder = join(paths.path2derivatives, 'design-matrices/{0}/{1}'.format(args.language, model_name))
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not
        name = os.path.basename(os.path.splitext(runs[i][0])[0])
        run_name = name.split('_')[-1] # extract the name of the run
        path2output = join(output_parent_folder, 'design-matrices_{0}_{1}_{2}'.format(args.language, model_name, run_name) + extension)

        if compute(path2output, args.overwrite):
            merge = pd.concat([pd.read_csv(path2features, header=None) for path2features in runs[i]], axis=1) # concatenate horizontaly the read csv files of a run
            merge.columns = models
            merge.to_csv(path2output, index=False)
