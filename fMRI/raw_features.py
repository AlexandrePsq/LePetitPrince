import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import argparse
from os.path import join
import importlib

import warnings
warnings.simplefilter(action='ignore')

from utilities.settings import Paths, Params
from utilities.utils import get_data, check_folder, compute, get_output_parent_folder, get_path2output
import pandas as pd
from joblib import Parallel, delayed


paths = Paths()
params = Params()


def compute_raw_features(module, run, output_parent_folder, input_data_type, output_data_type, language, model_name, model_category, extension, overwrite):
    model = module.load() # load an already trained model
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    path2output = get_path2output(output_parent_folder, output_data_type, language, model_name, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        textgrid = pd.read_csv(join(paths.path2data, input_data_type, language, model_category, 'onsets-offsets', '{}_{}_{}_onsets-offsets_{}'.format(input_data_type, language, model_category, run_name)+extension)) # df with onsets-offsets-word
        raw_features, columns2retrieve, save_all = module.generate(model, run, language, textgrid, overwrite) # generate raw_features from model's predictions

        # using offsets of words instead of onsets so that the subject has heard the word
        if save_all:
            raw_features['offsets'] = textgrid['offsets']
            raw_features['duration'] = 0
            raw_features.to_csv(save_all, index=False) # saving all raw_features

        columns2retrieve += ['offsets', 'duration']
        result = raw_features[columns2retrieve]
        if columns2retrieve == ['surprisal', 'offsets', 'duration']:
            result['offsets'] += params.pref.shift_surprisal
        result.to_csv(path2output, index=False) # saving raw_features.csv


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate raw features from raw data for a given model in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--language", type=str, default='english', help="Language of the text and the model.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use to generate the raw features.")
    parser.add_argument("--model_category", type=str, help="Category of the model to use to generate the raw features.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", default=False, action='store_true', help="Precise if we want to run code in parallel")


    args = parser.parse_args()

    extension = '.csv'
    output_data_type = 'raw-features'
    source = 'fMRI'
    model_name = args.model_name
    model_category = args.model_category

    if model_name.split('_')[0] in ['rms', 'f0']:
        input_data_type = 'wave'
    else:
        input_data_type = 'text'

    module = importlib.import_module("models.{}.{}".format(args.language, model_name)) # import the right module depending on the model we want to use
    
    output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_name)
    check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

    raw_data = get_data(args.language, input_data_type, model=model_category, source='fMRI') # retrieve the data to transform

    if not args.parallel:
        for i, run in enumerate(raw_data):
            compute_raw_features(module, run, output_parent_folder, input_data_type, output_data_type, args.language, model_name, model_category, extension, args.overwrite)
    else:
        Parallel(n_jobs=-2)(delayed(compute_raw_features) \
                    (module, run, output_parent_folder, input_data_type, output_data_type, args.language, model_name, model_category, extension, args.overwrite) for run in raw_data)


