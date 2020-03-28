from sklearn.model_selection import LeavePOut



class Splitter(object):
    """ Tools to split lists or groups into several folds.
    """

    def __init__(self, out_per_fold):
        """ Instanciation of Splitter class. We specify the number of runs
        to leave out for the test set.
        Arguments:
            - out_per_fold: int
        """
        self.out_per_fold = out_per_fold
        pass
    
    def split(self, X_train, Y_train, run_train=None, run_test=None):
        """ Split lists in differents folds for cross validation.
        Arguments:
            - X_train: list
            - Y_train: list
            - run_train: list
            - run_test: list
        Returns:
            - list (of dict)
        """
        result = []
        logo = LeavePOut(self.out_per_fold)
        for train, test in logo.split(X_train):
            y_train = [Y_train[i] for i in train]
            x_train = [X_train[i] for i in train]
            y_test = [Y_train[i] for i in test]
            x_test = [X_train[i] for i in test]
            result.append({'X_train': x_train,
                        'Y_train': y_train,
                        'X_test': x_test,
                        'Y_test': y_test,
                        'run_train': [run_train[index] for index in train] if run_train is not None else train,
                        'run_test': [run_train[index] for index in test] if run_train is not None else test
                        })
        return result
        
            
