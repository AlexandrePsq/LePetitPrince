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

def wordrate(iterator, language, path):
    return np.ones(len(iterator))

def content_words(iterator, language, path):
    function_words_list = get_function_words_list(language)
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 0 if iterator[index] not in function_words_list else 1
    return result

def function_words(iterator, language, path):
    function_words_list = get_function_words_list(language)
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 1 if iterator[index] in function_words_list else 0
    return result

def get_function_words_list(language):
    function_words_list = open(os.path.join(paths.path2code, 'models/{}/WORDRATE/'.format(language), 'function_words.txt'), 'r').read()
    return function_words_list.split('\n')

def log_word_freq(iterator, language, path):
    database = pd.read_csv(os.path.join(paths.path2data, 'text', language, 'lexique_database.tsv'), delimiter='\t')
    result = np.zeros(len(iterator))
    words = np.array(database['Word'])
    word_with_issues = {
        've': 'have',
        'hadn': 'had'
    }
    for index in range(len(iterator)):
        try:
            word = iterator[index]
            if word in word_with_issues:
                word = word_with_issues[word]
            print(np.argwhere(words==word))
            print(word)
            result[index] = database['Lg10WF'][np.argwhere(words==word)[0][0]]
        except:
            word = iterator[index].capitalize()
            print(np.argwhere(words==word.capitalize()))
            print(word)
            result[index] = database['Lg10WF'][np.argwhere(words==word.capitalize())[0][0]]
    return result
    
def word_position(iterator, language, path):
    # shitty function that is just relevant for this study (was hand made)
    name = os.path.basename(os.path.splitext(path)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    data = np.load(os.path.join(paths.path2data, 'text', language, 'word_position_{}.npy'.format(run_name)))
    return data