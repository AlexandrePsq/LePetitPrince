import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.append(root)

import warnings
warnings.simplefilter(action='ignore')

import torch
import os
import pandas as pd
import numpy as np

from utilities.settings import Params, Paths
from utilities.utils import check_folder



def load():
    from .LSTM import model, utils
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    mod = model.RNNModel('LSTM', 5, 600, 600, 3, dropout=0.2) # ntoken is chosen randomly, it will or has been determined during training
    data_name = 'wiki_kristina'
    language = 'english'
    return utils.load(mod, data_name, language)


def generate(model, run, language):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    model_name = 'lstm_wikikristina_embedding-size_{}_nhid_{}_nlayers_{}_dropout_{}'.format(model.param['ninp'], model.param['nhid'], model.param['nlayers'], str(model.param['dropout']).replace('.', ''))
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### parameters studied ####
    retrieve_surprisal = True
    #### generating raw-features ####
    if os.path.exists(path):
        raw_features = pd.read_csv(path)
    else:
        raw_features = model.generate(run, language)
        save_all = path
    #### Retrieving data of interest ####
    columns2retrieve = raw_features.columns
    if retrieve_surprisal:
        columns2retrieve.append('surprisal')
    return raw_features, columns2retrieve, save_all


if __name__ == '__main__':
    from LSTM import model, train
    params = Params()
    paths = Paths()
    mod = model.RNNModel('LSTM', 5, 600, 600, 3, dropout=0.2)
    data = os.path.join(paths.path2data, 'text', 'english', 'lstm_training')
    data_name = 'wiki_kristina'
    language = 'english'
    train.train(mod, data, data_name, language, eval_batch_size=params.pref.eval_batch_size, bsz=params.pref.bsz)