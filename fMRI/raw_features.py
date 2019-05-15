import argparse
from os.path import join
import glob
import importlib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ..utilities.settings import Paths

paths = Paths()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate raw features from original text for a given model in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")


    args = parser.parse_args()

    for model_ in args.models[0]:
        module = importlib.import_module("..models.{}.{}.py".format(args.language, model_)) # import the right module depending on the model we want to use
        model = module.create_model() # initialized the model
        model.load() # load an already trained model

        if model_ in ['rms', 'f0']:
            data_type = 'wave'
        else:
            data_type = 'text'
        data = get_data(args.language, data_type, model_, source='fMRI', args.test)

        for i, run in enumerate(data):
            raw_features = model.generate(run) # generate raw_features from model's predictions

            if len(list(raw_features.columns)) < 3:
                raw_features['duration'] = 0

            name = os.path.basename(run)
            name = name.split('_')[-1]
            path2output = join(paths.path2derivatives, 'raw_features/{0}/{1}/raw_features_{0}_{1}_{2}'.format(args.language, model_, name))
            raw_features.to_csv(path2output, index=False)