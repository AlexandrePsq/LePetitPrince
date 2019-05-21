import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import argparse
from os.path import join
import importlib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utilities.settings import Paths
from utilities.utils import *
import pandas as pd
from joblib import Parallel, delayed


paths = Paths()


def compute_raw_features(model, run, output_parent_folder, input_data_type, output_data_type, language, model_name, extension, overwrite):
    #run = run[0]
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    path2output = get_path2output(output_parent_folder, output_data_type, language, model_name, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        raw_features = model.generate(run) # generate raw_features from model's predictions

        if len(list(raw_features.columns)) < 3:
            raw_features['onsets'] = pd.read_csv(join(paths.path2data, input_data_type, language, 'onsets-offsets', '{}_{}_onsets-offsets_{}'.format(input_data_type, language, run_name)+extension))['onsets']
            raw_features['duration'] = 0

        raw_features.to_csv(path2output, index=False) # saving raw_features.csv


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate raw features from raw data for a given model in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we want to run code in parallel")


    args = parser.parse_args()

    extension = '.csv'
    output_data_type = 'raw_features'
    source = 'fMRI'
    model_name = args.model_name

    if model_name in ['rms', 'f0']:
        input_data_type = 'wave'
    else:
        input_data_type = 'text'

    module = importlib.import_module("models.{}.{}".format(args.language, model_name)) # import the right module depending on the model we want to use
    model = module.create_model() # initialized the model
    model.load() # load an already trained model

    output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_name)
    check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

    raw_data = get_data(args.language, input_data_type, model=model_name, source='fMRI') # retrieve the data to transform

    if not args.parallel:
        for i, run in enumerate(raw_data):
            compute_raw_features(model, run, output_parent_folder, input_data_type, output_data_type, args.language, model_name, extension, args.overwrite)
    else:
        Parallel(n_jobs=-2)(delayed(compute_raw_features) \
                    (model, run, output_parent_folder, input_data_type, output_data_type, args.language, model_name, extension, args.overwrite) for run in raw_data)


