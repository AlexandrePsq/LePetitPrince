import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import numpy as np
from utilities.settings import Params, Paths
from scipy.integrate import simps
import wave

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################

def rms(iterator, language, frame_rate):
    return np.apply_along_axis(lambda y: simps(y**2, dx=1/frame_rate),1, iterator)
