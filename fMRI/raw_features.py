import argparse
from os.path import join
import importlib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ..utilities.settings import Paths
from ..utilities.utils import *
import pandas as pd

paths = Paths()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate raw features from original text for a given model in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")


    args = parser.parse_args()

    extension = '.csv'
    output_data_type = 'raw_features'
    source = 'fMRI'

    for model_ in args.models[0]:
        if model_ in ['rms', 'f0']:
            input_data_type = 'wave'
        else:
            input_data_type = 'text'

        module = importlib.import_module("..models.{}.{}.py".format(args.language, model_)) # import the right module depending on the model we want to use
        model = module.create_model() # initialized the model
        model.load() # load an already trained model

        output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

        raw_data = get_data(args.language, input_data_type, model=model_, source='fMRI', test=args.test) # retrieve the data to transform

        for i, run in enumerate(raw_data):
            name = os.path.basename(os.path.splitext(run)[0])
            run_name = name.split('_')[-1] # extract the name of the run
            path2output = get_path2output(output_parent_folder, output_data_type, args.language, model_, run_name, extension)
            
            if compute(path2output, overwrite=args.overwrite):
                raw_features = model.generate(run) # generate raw_features from model's predictions

                if len(list(raw_features.columns)) < 3:
                    raw_features['onsets'] = pd.read_csv(join(paths.path2data, input_data_type, args.language, 'onsets-offsets', '{}_{}_onsets-offsets_{}'.format(input_data_type, args.language, run_name)+extension))['onsets']
                    raw_features['duration'] = 0

                raw_features.to_csv(path2output, index=False) # saving raw_features.csv
