import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.append(root)

import warnings
warnings.simplefilter(action='ignore')

import os
import pandas as pd


from utilities.settings import Paths
from utilities.utils import check_folder

paths = Paths()

def load():
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    from .PARSER import model
    from .PARSER.utils import bottomup, topdown
    language = 'english'
    mod = model.Parser([bottomup, topdown], language)
    return mod

def generate(mod, run, language, textgrid, overwrite=False):
    from .PARSER import model
    from .PARSER.utils import bottomup, topdown
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    mod = model.Parser([bottomup, topdown], language) # all functions
    model_name = 'parser_bottomup_topdown'
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### generating raw-features ####
    if (os.path.exists(path)) & (not overwrite):
        raw_features = pd.read_csv(path)
    else:
        raw_features = mod.generate(run, language, textgrid)
        save_all = path
    #### Retrieving data of interest ####
    columns2retrieve = [function.__name__ for function in mod.functions]
    textgrid = pd.read_csv(os.path.join(paths.path2data, 'text', language, 'PARSER', 'onsets-offsets', '{}_{}_{}_onsets-offsets_{}'.format('text', language, 'PARSER', run_name)+'.csv')) # df with onsets-offsets-word
    return raw_features[:textgrid.offsets.count()], columns2retrieve, save_all