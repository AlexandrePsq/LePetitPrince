import numpy as np

from sklearn.decomposition import PCA

from logger import Logger



class Compressor(object):
    """ Perform data compression over lists of numpy arrays.
    """
    
    def __init__(self, n_components):
        self.ncomponents = ncomponents
        pass

    def pca(self, X_list, X_test, logger=None):
        """ Classical PCA train on the concatenated set
        of matrices given in X_list.
        Arguments:
            - X_list: list
            - X_test: list
        """
        X_lengths = [m.shape[0] for m in X_list]
        X_all = np.vstack(X_list)
        pca = PCA(n_components=self.ncomponents)
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
