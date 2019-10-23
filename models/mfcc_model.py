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

paths =  Paths()

def load():
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    from .MFCC import model
    language = 'english'
    mod = model.MFCC(language)
    return mod

def generate(mod, run, language, textgrid, overwrite=False):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    model_name = 'mfcc_model'
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### generating raw-features ####
    if (os.path.exists(path)) & (not overwrite):
        raw_features = pd.read_csv(path)
    else:
        raw_features = mod.generate(run, language)
        save_all = path
    #### Retrieving data of interest ####
    columns2retrieve = ["mfcc #{}".format((i)//3) if i%3==0 else ("mfcc' #{}".format((i)//3) if i%3==1 else "mfcc'' #{}".format((i)//3)) for i in range(mod.num_cepstral*3)]
    textgrid = pd.read_csv(os.path.join(paths.path2data, 'wave', language, 'MFCC', 'onsets-offsets', '{}_{}_{}_onsets-offsets_{}'.format('wave', language, 'MFCC', run_name)+'.csv')) # df with onsets-offsets-word
    return raw_features[:textgrid.offsets.count()], columns2retrieve, save_all