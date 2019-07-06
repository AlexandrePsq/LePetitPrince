from utilities.settings import Params, Paths
import os
import itertools


paths = Paths()

layers_name = {'[0]':'first-layer',
                '[1]':'second-layer',
                '[2]':'third-layer',
                '[3]':'fourth-layer',
                '[4]':'fifth-layer',
                '[5]':'sixth-layer',
                '[6]':'seventh-layer',
                '[7]':'eighth-layer',
                '[8]':'ninth-layer',
                '[9]':'tenth-layer'
                }

language = 'english'
embedding_size_list = [600]
nhid_list = [600] # [200]
nlayers_list = [1,2,3] # [2,3]
dropout_list = [0.2]
parameters_list = [['hidden'], ['cell'], ['c_tilde'], ['in'], ['forget'], ['out'], []]

if __name__ == '__main__':

    for [embedding_size, nhid, nlayers, dropout, parameters] in [[ninp, nhid, nlay, drop, par] for ninp in embedding_size_list for nhid in nhid_list for nlay in nlayers_list for drop in dropout_list for par in parameters_list]:
        layer_range = [[item] for item in sorted(range(nlayers))]+[sorted(range(nlayers))] if len(sorted(range(nlayers))) > 1 else [[item] for item in sorted(range(nlayers))]
        for analyzed_layers in layer_range:

            retrieve_surprisal = (parameters==[]) # retrieve surprisal only if we do not retrieve the other

            extension = (parameters[0] if len(parameters)>0 else 'surprisal') + '_'
            extension += layers_name[str(analyzed_layers)] if str(analyzed_layers) in layers_name.keys() else 'all_layers'
            model_name = 'lstm_wikikristina_embedding-size_{}_nhid_{}_nlayers_{}_dropout_{}_'.format(embedding_size, nhid, nlayers, str(dropout).replace('.', '')) + extension

            if not os.path.isfile(model_name+'.py'):
                with open(os.path.join(paths.path2code, 'models', language, 'model_name.txt'), 'a+') as f:
                    f.write(model_name)
                    f.write('\n')

                result = \
                """
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
    mod = model.RNNModel('LSTM', 5, {0}, {1}, {2}, dropout={3}) # ntoken is chosen randomly, it will or has been determined during training
    data_name = 'wiki_kristina'
    language = 'english'
    return utils.load(mod, data_name, language)
                """.format(embedding_size, nhid, nlayers, dropout)

                result +=\
                """
def generate(model, run, language):
    name = os.path.basename(os.path.splitext(run)[0])
    run_name = name.split('_')[-1] # extract the name of the run
    save_all = None
    model_name = 'lstm_wikikristina_embedding-size_{}_nhid_{}_nlayers_{}_dropout_{}'.format(model.param['ninp'], model.param['nhid'], model.param['nlayers'], str(model.param['dropout']).replace('.', ''))
    check_folder(os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name))
    path = os.path.join(Paths().path2derivatives, 'fMRI/raw-features', language, model_name, 'raw-features_{}_{}_{}.csv'.format(language, model_name, run_name))
    #### parameters studied ####
                """

                result +=\
                """
    parameters = sorted({0})
    analyzed_layers = sorted({1}) # first layer
    retrieve_surprisal = {2}
                """.format(parameters, analyzed_layers, retrieve_surprisal)

                result +=\
                """
    #### generating raw-features ####
    if os.path.exists(path):
        raw_features = pd.read_csv(path)
    else:
        raw_features = model.generate(run, language)
        save_all = path
    #### Retrieving data of interest ####
    weight2retrieve = []
    for layer in analyzed_layers:
        weight2retrieve.append(np.arange(model.param['nhid']*layer, model.param['nhid']*(layer+1)))
    columns2retrieve = ['raw-{}-{}'.format(name, i) for i in np.hstack(weight2retrieve) for name in parameters]
    
    if retrieve_surprisal:
        columns2retrieve.append('surprisal')
    return raw_features, columns2retrieve, save_all
                """

                result +=\
                """

if __name__ == '__main__':
    from LSTM import model, train
    params = Params()
    paths = Paths()
    mod = model.RNNModel('LSTM', 5, {0}, {1}, {2}, dropout={3})
    data = os.path.join(paths.path2data, 'text', 'english', 'lstm_training')
    data_name = 'wiki_kristina'
    language = 'english'
    train.train(mod, data, data_name, language, eval_batch_size=params.pref.eval_batch_size, bsz=params.pref.bsz)
                """.format(embedding_size, nhid, nlayers, dropout)


                with open(os.path.join(paths.path2code, 'models', language, model_name + '.py'), 'w') as f:
                    f.write(result)