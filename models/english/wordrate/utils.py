import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import numpy as np
from utilities.settings import Params, Paths

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################

def wordrate(iterator):
    return np.ones(len(iterator))

def content_words(iterator):
    function_words_list = get_function_words_list()
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 0 if iterator[index] not in function_words_list else 1
    return result

def function_words(iterator):
    function_words_list = get_function_words_list()
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 1 if iterator[index] in function_words_list else 0
    return result

def get_function_words_list():
    function_words_list = open('function_words.txt', 'r').read()
    return function_words_list.split('\n')