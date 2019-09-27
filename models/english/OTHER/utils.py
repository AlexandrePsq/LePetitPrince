import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from utilities.settings import Params, Paths

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################


def sentence_onset(iterator, language, path):
    result = np.zeros(len(iterator))
    eof_symbol = ['.', '!', '?', '"', '(',  '{', '[', '“', '”', "..."]
    for index in range(len(iterator)):
        if iterator[index] in eof_symbol:
            if index+1 < len(result):
                result[index+1] = 1
    count=0
    indexes = []
    for index in range(len(result)):
        if result[index] == 1 and count > 0:
            indexes.append(index - 1)
            count += 1
        elif result[index] == 1: 
            count += 1
        else:
            count = 0
    result[indexes] = 0
    result[0] = 1
    return np.array(result)