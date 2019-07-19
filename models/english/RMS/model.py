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


    def generate(self, path, language, textgrid, slice_period):
        iterator, frame_rate, n_frames, slice_length = tokenize(path, slice_period=slice_period)

        parameters = {'frame_rate': frame_rate, 
                        'n_frames': n_frames, 
                        'slice_length': slice_length}
        # create specific onsets-offsets
        source = 'wave'
        model_category = 'RMS'
        for index in range(1,params.nb_runs + 1):
            file_name = '{}_{}_{}_run{}.wav'.format(source, language, model_category, index)
            wave_file = wave.open(os.path.join(paths.path2data, source, language, model_category, file_name), mode='r'
            length = int(parameters['n_frames']/parameters['frame_rate'] // slice_period)
            offsets = np.cumsum(np.ones(length) * slice_period)
            offsets = np.array([round(x, 3) for x in offsets])
            offsets = np.apply_along_axis(lambda y: round(y, 3), 0, offsets)
            onsets = np.hstack([np.zeros(1), offsets[:-1]])
            duration = np.zeros(length)
            df = pd.DataFrame({})
            df['onsets'] = onsets
            df['offsets'] = offsets
            df['duration'] = duration
            saving_in = '{}_{}_{}_{}_run{}.csv'.format(source, language, model_category, 'onsets-offsets', index)
            df.to_csv(os.path.join(paths.path2data, source, language, model_category, 'onsets-offsets', saving_in), index=False)
        

        dataframes = [pd.DataFrame(function(iterator, language, parameters), columns=[function.__name__]) for function in self.functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result
