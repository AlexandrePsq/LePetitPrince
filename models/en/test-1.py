##############################
# this is a test model
#  (a dummy one)
##############################

import random
import pandas as pd
import pickle
import numpy as np


class Dummy(object):
    def __init__(self):
        self.vocab = []
        self.nb_features = 1
        self.name = 'test'

    def train(self, file):
        text = open(file, 'r').read()
        vocab = set(text.split())
        self.vocab  = vocab
        self.n = len(vocab)

    def save(self, path):
        with open(path, 'wb') as output:
            pickle.dump(self.vocab, output)

    def load(self, path='/Users/alexpsq/Code/NeuroSpin/LePetitPrince/derivatives/to_delete.csv'):
        with open(path, 'rb') as input:
            self.vocab = pickle.load(input)
            self.n = len(self.vocab)

    def generate(self, path):
        text = open(path, 'r').read()
        words = text.split()
        result = []
        for _ in words:
            # result.append(random.randint(0, self.n)/self.n)
            # result.append(0)
            result.append(np.random.randint(self.n, size=(self.nb_features))/self.n)
        headers = ['{}-amplitude{}'.format(self.name, i) for i in range(len(result[0]))]
        return pd.DataFrame(result, columns=headers)



def create_model():
    return Dummy()
