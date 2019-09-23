################################################################
# Energy of the signal sampled at slicing_period (10e-2 s)
#
################################################################
import sys
import os
import scipy.io.wavfile as wav
import speechpy


root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.append(root)


import pandas as pd
import numpy as np
from utilities.settings import Params, Paths

params = Params()
paths = Paths()



class MFCC(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, language, num_cepstral=13, frame_length=0.02):
        super(MFCC, self).__init__()
        self.language = language
        self.num_cepstral = num_cepstral
        self.frame_length = frame_length
    

    def __name__(self):
        return 'MFCC - {} features'.format(self.num_cepstral)


    def generate(self, path, language):
        fs, signal = wav.read(path)
        # no overlapping
        mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=self.frame_length,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, num_cepstral=self.num_cepstral)
        # create specific onsets-offsets
        source = 'wave'
        model_category = 'RMS'
        for index in range(1,params.nb_runs + 1):
            length = int(mfcc.shape[0])
            offsets = np.cumsum(np.ones(length) * self.frame_length)
            offsets = np.array([round(x, 3) for x in offsets])
            onsets = np.hstack([np.zeros(1), offsets[:-1]])
            duration = np.zeros(length)
            df = pd.DataFrame({})
            df['onsets'] = onsets
            df['offsets'] = offsets
            df['duration'] = duration
            saving_in = '{}_{}_{}_{}_run{}.csv'.format(source, language, model_category, 'onsets-offsets', index)
            df.to_csv(os.path.join(paths.path2data, source, language, model_category, 'onsets-offsets', saving_in), index=False)
        

        result = pd.DataFrame(mfcc, columns=['feature #{}'.format(i+1) for i in range(self.num_cepstral)])
        return result