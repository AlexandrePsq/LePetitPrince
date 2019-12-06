import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import math
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utilities.settings import Paths
from .tokenizer import tokenize
from gensim.models import Word2Vec
import pickle

paths = Paths()




###############################################################################
# Training code
###############################################################################



def train(model, path2data, data_name, language):
    print(model)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('Entering training...')
        sentences = tokenize(path2data, language, train=True)
        model = Word2Vec(sentences, min_count=1, size=model.param['embedding-size'], workers=8)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    path = '_'.join([model.__name__(), data_name, language]) + '.pt'
    path = os.path.join(paths.path2derivatives, 'fMRI/models', language, path)
    model.save(path)


