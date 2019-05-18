from sklearn.model_selection import KFold


class Splitter(KFold):
    
    def __init__(self, indexes_dict={}, n_splits=9, shuffle=False,
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
        for train_index, test_index in super().split(X):
            index_train = []
            index_test = []
            for i in train_index:
                beg, end = self.indexes_dict['run{}'.format(i+1)]
                index_train.append(np.arange(beg, end))
            list_indexes_train = np.hstack(index_train)
            for i in test_index:
                beg, end = self.indexes_dict['run{}'.format(i+1)]
                index_test.append(np.arange(beg, end))
            list_indexes_test = np.hstack(index_test)
            yield list_indexes_train, list_indexes_test
        
