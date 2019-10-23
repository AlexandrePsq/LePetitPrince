################################################################
# General Language Model including :
#   - sentence onset
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



class Other(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, functions, language):
        super(Other, self).__init__()
        self.functions = functions
        self.language = language

    def __name__(self):
        return '_'.join([function.__name__ for function in self.functions])

    def generate(self, path, language, textgrid):
        # iterator = tokenize(path, language) 
        iterator_with_ponctuation = tokenize(path, language) 
        dataframes = [pd.DataFrame(function(iterator_with_ponctuation, language, path), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result
