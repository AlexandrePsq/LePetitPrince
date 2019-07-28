import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utilities.settings import Params, Paths
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################


def neighborhood_density(self, method='mean', threshold=0.7):
    result = np.zeros(len(self.model.vocab))
    tmp = np.zeros((len(self.model.vocab), len(self.model.vocab)))
    for i in range(len(self.model.vocab) - 1):
        for j in range(i + 1, len(self.model.vocab)):
            tmp[i,j] = cosine_similarity(self.model.vectors[i], self.model.vectors[j])
    if method == 'mean':
        j
    elif method == 'threshold':
        tmp[tmp < threshold] = 0
