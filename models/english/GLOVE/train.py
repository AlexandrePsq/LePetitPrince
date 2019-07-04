import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


import math
import pandas as pd
import numpy as np
import time
from .data import Corpus
from tqdm import tqdm
from utilities.settings import Params, Paths
from .tokenizer import tokenize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

params = Params()
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
        model = Word2Vec(sentences, min_count=1, size=self.param['embedding-size'])
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

