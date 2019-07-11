################################################################
# Energy of the signal sampled at slicing_period (10e-2 s)
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
import wave

params = Params()
paths = Paths()



class EnergySpectrum(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, functions, language):
        super(EnergySpectrum, self).__init__()
        self.functions = functions
        self.language = language
    

    def __name__(self):
        return '_'.join([function.__name__ for function in self.functions])


    def generate(self, path, language, textgrid, slice_period=10e-3):
        iterator, frame_rate, n_frames, slice_length = tokenize(path, slice_period=slice_period)

        parameters = {'frame_rate': frame_rate, 
                        'n_frames': n_frames, 
                        'slice_length': slice_length}

        dataframes = [pd.DataFrame(function(iterator, language, parameters), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result
