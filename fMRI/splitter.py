from sklearn.model_selection import LeavePOut



class Splitter(object):
    """ Tools to split lists or groups into several folds.
    """

    def __init__(self, out_per_fold):
        self.out_per_fold = out_per_fold
        pass
    
    def split(X_list, Y_list):
        """ Split lists in differents folds for cross validation.
        Arguments:
            - X_list: list
            - Y_list: list
        """
        result = []
        logo = LeavePOut(self.out_per_fold)
        for train, valid in logo.split(X_list):
            y_train = [Y_list[i] for i in train]
            x_train = [X_list[i] for i in train]
            y_test = [Y_list[i] for i in valid]
            x_test = [X_list[i] for i in valid]
            result.append({'X_list': x_train,
                        'Y_list': y_train,
                        'X_test': x_test,
                        'Y_test': y_test
                        })
        return result
        
            
