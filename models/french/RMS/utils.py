import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import numpy as np
from utilities.settings import Params, Paths
from scipy.integrate import simps
from scipy.signal import hilbert
import wave

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################

def rms(iterator, language, parameters):
    result = np.apply_along_axis(lambda y: np.sqrt(np.mean(np.square(y, dtype=np.float64))),1, iterator)
    return result

    #frame_rate = parameters['frame_rate']
    #return np.apply_along_axis(lambda y: simps(np.abs(hilbert(y)), dx=1/frame_rate),1, iterator)