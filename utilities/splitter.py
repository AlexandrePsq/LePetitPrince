from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import LeaveOneOut
from .settings import Params

params = Params()

class Splitter(KFold):

    def __init__(self, indexes_dict={}, n_splits=params.nb_runs shuffle=False,
                 random_state=None):
        n_splits = n_splits
        self.indexes_dict = indexes_dict
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_runs, n_samples, n_features)
            Training data, where n_runs is the number of runs, n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        logo = LeaveOneOut()
        keys = list(self.indexes_dict.keys())
        for train_index, test_index in logo.split(keys):
            index_train = []
            index_test = []
            for i in train_index:
                beg, end = self.indexes_dict[keys[i]]
                index_train.append(np.arange(beg, end))
            list_indexes_train = np.hstack(index_train)
            for i in test_index:
                beg, end = self.indexes_dict[keys[i]]
                index_test.append(np.arange(beg, end))
            list_indexes_test = np.hstack(index_test)
            yield list_indexes_train, list_indexes_test
