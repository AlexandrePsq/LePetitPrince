import os
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge



class EncodingModel(object):
    """ General class for encoding models: fit an array
    of regressors to fMRI data.
    """

    def __init__(self, model=Ridge(), alpha=None):
        """ Instanciation of EncodingModel class.
        Arguments:
            - model: sklearn.linear_model
            - alpha: int
        """
        self.alpha = alpha # regularization parameter
        self.model = model
        self.model.set_params(alpha=alpha)
        pass
    
    def fit(self, X_list, Y_list):
        """ Fit the model for a given set of runs.
        Arguments:
            - X_list: list (of array)
            - Y_list: list (of array)
        """
        dm = np.vstack(X_list)
        fmri = np.vstack(Y_list)
        self.model = self.model.fit(dm,fmri)
    
    def predict(self, X):
        """ Compute the predictions of the model for a given
        input X.
        Arguments:
            - X: array
        """
        predictions = self.model.predict(X)
        return predictions
    
    def get_R2_coeff(self, predictions, Y):
        """ Compute the R2 score for each voxel (=list).
        Arguments:
            - X: array
            - Y: array
        """
        r2 = r2_score(Y, predictions, multioutput='raw_values')
        return r2
    
    def get_Pearson_coeff(self, predictions, Y):
        """ Compute the Pearson correlation coefficients
        score for each voxel (=list).
        Arguments:
            - X: array
            - Y: array
        """
        pearson_corr = [pearsonr(Y[:,i], predictions[:,i])[0] for i in range(Y.shape[1])]
        return pearson_corr
