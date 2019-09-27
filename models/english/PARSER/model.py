################################################################
# General Language Model Parsers
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
from utilities.settings import Params, Paths

params = Params()
paths = Paths()

class Parser(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, functions, language):
        super(Parser, self).__init__()
        self.functions = functions
        self.language = language


    def __name__(self):
        return '_'.join([function.__name__ for function in self.functions])


    def generate(self, path, language, textgrid):
        # iterator = tokenize(path, language) 
        iterator_with_ponctuation = tokenize(path, language) 
        dataframes = [pd.DataFrame(function(iterator_with_ponctuation, language, path), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        # create specific onsets-offsets
        source = 'text'
        model_category = 'PARSER'
        name = os.path.basename(os.path.splitext(path)[0])
        run_name = name.split('_')[-1] # extract the name of the run
        run_nb = run_name[-1]
        data = pd.read_csv(os.path.join(paths.path2data, 'text', language, 'PARSER', '{}_topdown.csv'.format(run_nb)))
        df = pd.DataFrame({})
        df['offsets'] = data['onset'].values # the formerly saved onset is in fact the offset...
        df['onsets'] = np.nan
        df['duration'] = 0
        saving_in = '{}_{}_{}_{}_{}.csv'.format(source, language, model_category, 'onsets-offsets', run_name)
        df.to_csv(os.path.join(paths.path2data, source, language, model_category, 'onsets-offsets', saving_in), index=False)
        return result
