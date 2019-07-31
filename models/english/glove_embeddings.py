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



def load():
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    from .GLOVE import model
    from .GLOVE.utils import embeddings
    language = 'english'
    #### parameters studied ####
    parameters = sorted([embeddings])
    mod = model.Glove(functions=parameters, language=language)
    return mod

def generate(mod, run, language, textgrid, overwrite=False):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    
    # Defining paths
    model_name = 'glove_model'
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### generating raw-features ####
    if (os.path.exists(path)) & (not overwrite):
        raw_features = pd.read_csv(path)
    else:
        raw_features = mod.generate(run, language, textgrid)
        save_all = path
    #### Retrieving data of interest ####
    # columns2retrieve = [function.__name__ for function in mod.functions]
    columns2retrieve = ['embedding-{}'.format(index) for index in range(mod.param['embedding-size'])]
    return raw_features[:textgrid.offsets.count()], columns2retrieve, save_all