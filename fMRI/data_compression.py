"""
General framework regrouping various compression methods to be performed on the data.
===================================================
A Compressor instanciation requires:
    - n_components_list: list regrouping the number of components to keep for each model
    representations dimension reduction,
    - indexes: list of numpy arrays listing the columns indexes for each model to retrieve the right
    columns when compressing specific model representations,
    - compression_types: list of string regrouping the specific data reduction methods to apply to 
    each model representations.
This class allows to compress differently specific parts of a list of matrices.
"""



import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils import clean_nan_rows



class Compressor(object):
    """ Perform data compression over lists of numpy arrays.
    """
    
    def __init__(self, n_components_list, indexes, compression_types):
        """ Instanciation of the class Compressor.
        Arguments:
            - n_components_list: list (of list)
            - indexes: list (of list)
            - compression_types: list (of str)
        """
        self.ncomponents_list = n_components_list
        self.indexes = indexes
        self.compression_types = compression_types
        self.bucket = []
        pass
        
    def clean_bucket(self):
        """ Clean instance bucket."""
        self.bucket = []
    
    def identity(self, X_train, X_test, n_components=None):
        """ Identity function.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        return {'X_train': X_train, 'X_test': X_test}

    def pca(self, X_train, X_test, n_components):
        """ Classical PCA train on the concatenated set
        of matrices given in X_train.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        X_lengths = [m.shape[0] for m in X_train]
        X_all = np.vstack(X_train)
        pca = PCA(n_components=n_components)
        pca.fit(X_all)
        X_all = pca.transform(X_all)
        index = 0
        X_train_ = []
        X_test_ = []
        for i in X_lengths:
            X_train_.append(pca.transform(X_all[index:index+i,:]))
            index += i
        for matrix in X_test:
            X_test_.append(pca.transform(matrix))
        return {'X_train': X_train_, 'X_test': X_test_}

    def compress(self, X_train, X_test):
        """ Compress the data with different compression methods
        for the different representations coming from different models.
        Arguments:
            - X_train: list
            - X_test: list
        """
        for index, indexes in enumerate(self.indexes):
            func = getattr(self, self.compression_types[index])
            X_train_ = [clean_nan_rows(X[:,indexes]) for X in X_train]
            X_test_ = [clean_nan_rows(X[:,indexes]) for X in X_test]
            self.bucket.append(func(X_train_, X_test_, self.ncomponents_list[index]))
        
        X_train = [pd.concat([pd.DataFrame(data['X_train'][run_index]) for data in self.bucket], axis=1).values for run_index in range(len(self.bucket[0]['X_train']))]
        X_test = [pd.concat([pd.DataFrame(data['X_test'][run_index]) for data in self.bucket], axis=1).values for run_index in range(len(self.bucket[0]['X_test']))]
        self.clean_bucket()
        return {'X_train': X_train, 'X_test': X_test}
