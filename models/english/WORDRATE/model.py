################################################################
# General Language Model including :
#   - default wordrate raw-features
#   - wordrate depending on specified criteria
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



class Wordrate(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, functions, language):
        super(Wordrate, self).__init__()
        self.functions = functions
        self.language = language

    def __name__(self):
        return '_'.join([function.__name__ for function in self.functions])

    def generate(self, path, language, textgrid):
        # iterator = tokenize(path, language) 
        iterator = list(textgrid['word']) # we suppose the textgrid dataframe (=csv file with onsets and offsets issue from original textgrid) has been created thanks to the tokennize function
        dataframes = [pd.DataFrame(function(iterator, language, path), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result
