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
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA 
from sklearn import manifold
import umap

from utils import clean_nan_rows



class Compressor(object):
    """ Perform data compression over lists of numpy arrays.
    """
    
    def __init__(self, n_components_list, indexes, compression_types, manifold_method=None, manifold_args=None):
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
        self.manifold_method = manifold_method
        self.manifold_args = manifold_args
        
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
        """ Classical PCA trained on the concatenated set
        of matrices given in X_train.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        X_lengths = [m.shape[0] for m in X_train]
        X_all = np.vstack(X_train)
        pca_ = PCA(n_components=n_components)
        pca_.fit(X_all)
        index = 0
        X_train_ = []
        X_test_ = []
        for i in X_lengths:
            X_train_.append(pca_.transform(X_all[index:index+i,:]))
            index += i
        for matrix in X_test:
            X_test_.append(pca_.transform(matrix))
        return {'X_train': X_train_, 'X_test': X_test_}

    def ica(self, X_train, X_test, n_components, random_state=1111):
        """ Classical ICA trained on the concatenated set
        of matrices given in X_train.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        X_lengths = [m.shape[0] for m in X_train]
        X_all = np.vstack(X_train)
        ica_ = FastICA(n_components=n_components, random_state=random_state)
        ica_.fit(X_all)
        index = 0
        X_train_ = []
        X_test_ = []
        for i in X_lengths:
            X_train_.append(ica_.transform(X_all[index:index+i,:]))
            index += i
        for matrix in X_test:
            X_test_.append(ica_.transform(matrix))
        return {'X_train': X_train_, 'X_test': X_test_}
    
    def manifold_reduction(self, X_train, X_test, n_components):
        """ Classical PCA trained on the concatenated set
        of matrices given in X_train.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        reduction_method = getattr(manifold, self.manifold_method)
        reductor = reduction_method(**self.manifold_args)
        reductor.set_params(n_components=n_components)
        if hasattr(reduction_method, 'transform'):
            X_lengths = [m.shape[0] for m in X_train]
            X_all = np.vstack(X_train)
            reductor.fit(X_all)
            index = 0
            X_train_ = []
            X_test_ = []
            for i in X_lengths:
                X_train_.append(reductor.transform(X_all[index:index+i,:]))
                index += i
            for matrix in X_test:
                X_test_.append(reductor.transform(matrix))
        else:
            X_lengths = [m.shape[0] for m in X_train] + [m.shape[0] for m in X_test]
            X_all = np.vstack(X_train + X_test)
            results = reductor.fit_transform(X_all)
            index = 0
            test_size = len(X_test)
            X_train_ = []
            X_test_ = []
            print(i)
            for i in X_lengths[:-test_size]:
                X_train_.append(results[index:index+i,:])
                index += i
                print(i)
            for i in X_lengths[-test_size:]:
                X_test_.append(results[index:index+i,:])
                index += i
                print(i)
        return {'X_train': X_train_, 'X_test': X_test_}

    def umap(self, X_train, X_test, n_components, random_state=1111):
        """ Classical ICA trained on the concatenated set
        of matrices given in X_train.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        X_lengths = [m.shape[0] for m in X_train]
        X_all = np.vstack(X_train)
        umap_ = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=n_components)
        umap_.fit(X_all)
        index = 0
        X_train_ = []
        X_test_ = []
        for i in X_lengths:
            X_train_.append(umap_.transform(X_all[index:index+i,:]))
            index += i
        for matrix in X_test:
            X_test_.append(umap_.transform(matrix))
        return {'X_train': X_train_, 'X_test': X_test_}
    
    def similarity_cluster(self, X_train, X_test, n_components):
        """ Average most similar features. Trained on the concatenated set
        of matrices given in X_train. Details: cluster features according to their L2 norm
        and average features among a given cluster.
        Arguments:
            - X_train: list
            - X_test: list
            - n_components: int
        """
        X_lengths = [m.shape[0] for m in X_train]
        X_all = np.vstack(X_train)
        if "linkage" not in self.manifold_args:
            self.manifold_args["linkage"] = "ward"
        sc = AgglomerativeClustering(n_clusters=n_components, **self.manifold_args)
        sc.fit(X_all.T)
        index = 0
        X_train_ = []
        X_test_ = []
        for i in X_lengths:
            result = np.zeros((X_all[index:index+i,:].shape[0], n_components))
            for component in range(n_components):
                result[:, component] = np.mean(X_all[index:index+i,:][:, sc.labels_==component], axis=1) 
            X_train_.append(result)
            index += i
        for matrix in X_test:
            result = np.zeros((matrix.shape[0], n_components))
            for component in range(n_components):
                result[:, component] = np.mean(matrix[:, sc.labels_==component], axis=1) 
            X_test_.append(result)
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
