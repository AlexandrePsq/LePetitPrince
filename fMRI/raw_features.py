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
from utilities.utils import get_data, compute, get_path2output, standardization
import pandas as pd
from joblib import Parallel, delayed


paths = Paths()
params = Params()


def compute_raw_features(run, source, data_type, step, language, model_name, model_category, extension, overwrite):
    """Compute raw-features (model predictions) from input data.
    - First: the model of interest is loaded as a module.
    - Second: We generate the raw-features.
    - Third: We generate the text-grid if not specified by default.
    - Fourth: We save the data if we did not extract it from former computations.
    :run: (str) path to the raw-feature file.
    :source: (str) Source of acquisition used (fMRI or MEG).
    :data_type: (str) type of the input data that is being processed.
    :step: (str) step of the pipeline that is being computed.
    :language: (str) language used.
    :model_name: (str) name of the model being studied.
    :model_category: (str) category of the model being studied.
    :extension: (str) extension used for the derivatives that will be saved.
    :overwrite: (boolean) for overwriting existing data.
    """
    module = importlib.import_module("models.{}".format(model_name)) # import the right module depending on the model we want to use
    model = module.load() # load an already trained model
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    path2output = get_path2output(source, step, language, model_name, run_name, extension)
    
    if compute(path2output, overwrite=overwrite):
        try:
            textgrid = pd.read_csv(join(paths.path2data, data_type, language, model_category, 'onsets-offsets', '{}_{}_{}_onsets-offsets_{}'.format(data_type, language, model_category, run_name)+extension)) # df with onsets-offsets-word
        except:
            # Some models will sample at their own pace, and therefore will compute the textgrid during raw-features generation
            textgrid = None
            print('WARNING: textgrid still not computed.')
        raw_features, columns2retrieve, save_all = module.generate(model, run, language, textgrid, overwrite) # generate raw_features from model's predictions
        if textgrid is None:
            textgrid = pd.read_csv(join(paths.path2data, data_type, language, model_category, 'onsets-offsets', '{}_{}_{}_onsets-offsets_{}'.format(data_type, language, model_category, run_name)+extension))
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


    # Retrieving data
    extension = '.csv'
    step = 'raw-features'
    source = 'fMRI'
    model_name = args.model_name
    model_category = args.model_category

    if model_name.split('_')[0] in ['rms', 'f0', 'mfcc']:
        input_data_type = 'wave'
    else:
        input_data_type = 'text'
    
    raw_data = get_data(args.language, input_data_type, model=model_category, source='fMRI') # retrieve the data to transform


    # Computing Raw-features
    if not args.parallel:
        for i, run in enumerate(raw_data):
            compute_raw_features(run, source, input_data_type, step, args.language, model_name, model_category, extension, args.overwrite)
    else:
        Parallel(n_jobs=-2)(delayed(compute_raw_features) \
                    (run, source, input_data_type, step, args.language, model_name, model_category, extension, args.overwrite) for run in raw_data)
    
    if params.pca:
        data = [pd.read_csv(run) for run in get_data(args.language, step, model=model_name, source='fMRI')]
        matrices = [df.values for df in data]
        matrices = standardization(matrices, model_name, pca_components=300, scale=False)
        for i in range(len(matrices)):
            run_name = 'run{}'.format(i)
            path2output = get_path2output(source, step, args.language, model_name, run_name, extension)

            result = pd.DataFrame(matrices[i], columns=['{}_component-{}'.format(model_name, index) for index in range(params.n_components_default)])
            result.to_csv(path2output, index=False)
