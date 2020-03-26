import numpy as np

from sklearn.decomposition import PCA

from logger import Logger



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
        self.ncomponents_list = ncomponents_list
        self.indexes = indexes
        self.compression_types = compression_types
        self.bucket = []
        pass
        
    def clean_bucket(self):
        self.bucket = []

    def pca(self, X_list, X_test, n_components, logger=None):
        """ Classical PCA train on the concatenated set
        of matrices given in X_list.
        Arguments:
            - X_list: list
            - X_test: list
            - logger: Logger
        """
        X_lengths = [m.shape[0] for m in X_list]
        X_all = np.vstack(X_list)
        pca = PCA(n_components=n_components)
        pca.fit(X_all)
        if logger:
            logger.figure(np.cumsum(pca.explained_variance_ratio_))
        X_all = pca.transform(X_all)
        index = 0
        X_list_ = []
        X_test_ = []
        for i in lengths:
            X_list_.append(pca.transform(X_all[index:index+i,:]))
            index += i
        for matrix in X_test:
            X_test_.append(pca.transform(matrix))
        return {'X_list': X_list_, 'X_test': X_test_}

    def compress(self, X_list, X_test, logger=None):
        """ Compress the data with different compression methods
        for the different representations coming from different models.
        Arguments:
            - X_list: list
            - X_test: list
            - logger: Logger
        """
        for index, indexes in enumerate(self.indexes):
            func = getattr(self, self.compression_types[index])
            X_list_ = X_list[:,indexes]
            X_test_ = X_test[:,indexes]
            self.bucket.append(func(X_list_, X_test_, self.ncomponents_list[index], logger)
        
        X_list = np.hstack([data['X_list'] for data in self.bucket]
        X_test = np.hstack([data['X_test'] for data in self.bucket]
        self.clean_bucket()
        return {'X_list': X_list, 'X_test': X_test}
