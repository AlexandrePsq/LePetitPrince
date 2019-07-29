import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utilities.settings import Params, Paths
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

params = Params()
paths = Paths()


###############################################################################
# Utilitiy functions
###############################################################################

def concat(l, m):
    # concatenate two vectors 
    if l.shape != m.T.shape:
        raise ValueError('Dimension mismatch {} != {}'.format(l.shape, m.T.shape))
    elif ((l.shape[0] != 1) and (l.shape[1] != 1)) :
        raise ValueError('Argument of shape ({}, {}) not a vector. '.format(l.shape[0], l.shape[1]))
    elif ((m.shape[0] != 1) and (m.shape[1] != 1)) :
        raise ValueError('Argument of shape ({}, {}) not a vector. '.format(m.shape[0], m.shape[1]))
    m =  m.T
    return np.hstack([l.reshape(1, -1), m.reshape(1, -1)])


###############################################################################
# Functions
###############################################################################


def neighborhood_density(self, iterator, language, method='mean', threshold=0.7):
    columns_activations = ['neighborhood_density']
    activations = []
    skip = 0
    # computing metric
    result = np.zeros(len(self.model.vocab))
    tmp = np.zeros((len(self.model.vocab), len(self.model.vocab)))
    for i in range(len(self.model.vocab) - 1):
        for j in range(i + 1, len(self.model.vocab)):
            tmp[i,j] = cosine_similarity(self.model.vectors[i], self.model.vectors[j])
        vector = tmp[0,1:] if i==0 else (concat(tmp[i,i+1:], tmp[:i,i]))
        if method == 'mean':
            result[i] = np.mean(vector)
        elif method == 'threshold':
            vector[vector < threshold] = 0
            result[i] = np.count_nonzero(vector)
    # generating prediction
    for item in tqdm(iterator):
        if item in self.words2add.keys():
            for word in self.words2add[item][0]:
                activations.append(result[self.model.vocab[word].index])
            skip = self.words2add[item][1]
        elif skip ==0:
            activations.append(result[self.model.vocab[word].index])
        else:
            skip -= 1
    return pd.DataFrame(np.vstack(activations), columns=columns_activations)


def embeddings(self, iterator, language):
    columns_activations = ['embedding-{}'.format(i) for i in range(self.param['embedding-size'])]
    activations = []
    skip = 0
    for item in tqdm(iterator):
        if item in self.words2add.keys():
            for word in self.words2add[item][0]:
                activations.append(self.model.vectors[self.model.vocab[word]])
            skip = self.words2add[item][1]
        elif skip ==0:
            activations.append(self.model.vectors[self.model.vocab[word]])
        else:
            skip -= 1
    return pd.DataFrame(np.vstack(activations), columns=columns_activations)