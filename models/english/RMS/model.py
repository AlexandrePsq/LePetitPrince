################################################################
# Energy of the signal for each word 
#
################################################################
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.append(root)


import pandas as pd
import numpy as np
from .tokenizer import tokenize
from .utils import *
from utilities.settings import Params, Paths
from . import utils


params = Params()
paths = Paths()



class EnergySpectrum(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, functions, language):
        super(EnergySpectrum, self).__init__()
        self.functions = functions
        self.language = language

    def generate(self, path, language):
        iterator = tokenize(path, language)
        name = os.path.basename(os.path.splitext(run)[0])
        run_name = name.split('_')[-1]
        model_category = os.path.basename(os.path.dirname(path))

        onset_path = os.path.join(paths.paths2data, 'text', 'english', model_category, 'onsets-offsets', 'text_{}_{}_onsets-offsets_{}'.format(language, model_category, run_name))
        onsets = pd.read_csv(onset_path)
        dataframes = [pd.DataFrame(function(iterator), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result
