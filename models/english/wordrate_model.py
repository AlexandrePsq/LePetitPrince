import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.append(root)

import warnings
warnings.simplefilter(action='ignore')

from .WORDRATE import model
from .WORDRATE.utils import wordrate, function_words, content_words
import os
import pandas as pd


from utilities.settings import Paths
from utilities.utils import check_folder



def load():
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    language = 'english'
    mod = model.Wordrate([wordrate], language)
    return mod

def generate(mod, run, language):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    mod = model.Wordrate([content_words, function_words, wordrate], language) # all functions
    model_name = 'wordrate_model'
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### parameters studied ####
    parameters = sorted([wordrate])
    #### generating raw-features ####
    if os.path.exists(path):
        raw_features = pd.read_csv(path)
    else:
        raw_features = mod.generate(run, language)
        save_all = path
    #### Retrieving data of interest ####
    columns2retrieve = [function.__name__ for function in model.Wordrate(parameters, language).functions]
    return raw_features, columns2retrieve, save_all