import os
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge



class EncodingModel(object):
    """ General class for encoding models: fit an array
    of regressors to fMRI data.
    """

    def __init__(self, model=Ridge(), alpha=None, alpha_min_log_scale=2, alpha_max_log_scale=4, nb_alphas=25, optimizing_criteria='R2'):
        """ Instanciation of EncodingModel class.
        Arguments:
            - model: sklearn.linear_model
            - alpha: int
            - alpha_min_log_scale: int
            - alpha_max_log_scale: int
            - nb_alphas: int
            - optimizing_criteria, str
        """
        self.alpha = alpha # regularization parameter
        self.model = model
        self.optimizing_criteria = optimizing_criteria
        self.alpha_list = [round(tmp, 5) for tmp in np.logspace(alpha_min_log_scale, alpha_max_log_scale, nb_alphas)]
    
    def fit(self, X_train, Y_train, alpha):
        """ Fit the model for a given set of runs.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
        """
        self.model.set_params(alpha=alpha)
        dm = np.vstack(X_train)
        fmri = np.vstack(Y_train)
        self.model = self.model.fit(dm,fmri)
    
    def predict(self, X_test):
        """ Compute the predictions of the model for a given
        input X_test.
        Arguments:
            - X_test: np.array
        Returns:
            - predictions: np.array
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def grid_search(self, X_train, Y_train, X_test, Y_test):
        """ Fit a model on the whole brain and return R2 coefficients,
        Pearson coefficients and regularization parameters.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - X_test: list (of np.array)
            - Y_test: list (of np.array)
        Returns:
            - result: dict
        """
        result = {'R2': [],
                    'Pearson_coeff': [],
                    'alpha': self.alpha_list
                    }
        for alpha in self.alpha_list:
            self.fit(X_train, Y_train, alpha)
            X_test = np.vstack(X_test)
            Y_test = np.vstack(Y_test)
            predictions = self.predict(X_test)
            result['R2'].append(self.get_R2_coeff(predictions, Y_test))
            result['Pearson_coeff'].append(self.get_Pearson_coeff(predictions, Y_test))
        result['R2'] = np.stack(result['R2'], axis=0)
        result['Pearson_coeff'] = np.stack(result['Pearson_coeff'], axis=0)
        return result
        
    def optimize_alpha(self, data, hyperparameter):
        """ Optimize the hyperparameter of a model given a
        list of measures.
        Arguments:
            - data: np.array (3D)
            - hyperparameter: list (of int)
        Returns:
            - voxel2alpha: list (of int)
            - alpha2voxel: dict (of list)
        """
        best_alphas_indexes = np.argmax(np.mean(data, axis=0), axis=0)
        voxel2alpha = np.array([hyperparameter[i] for i in best_alphas_indexes])
        alpha2voxel = {key:[] for key in hyperparameter}
        for index in range(len(voxel2alpha)):
            alpha2voxel[voxel2alpha[index]].append(index)
        return voxel2alpha, alpha2voxel
    
    def evaluate(self, X_train, X_test, Y_train, Y_test, R2, Pearson_coeff, alpha):
        """ Fit a model for each voxel given the parameter optimizing a measure.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - X_test: list (of np.array)
            - Y_test: list (of np.array)
            - R2: np.array (2D)
            - Pearson_coeff: np.array (2D)
            - alpha: np.array (1D)
        Returns:
            - result: dict
        """
        R2 = np.zeros((Y_test[0].shape[1]))
        Pearson_coeff = np.zeros((Y_test[0].shape[1]))
        x_test = np.vstack(X_test)
        data = R2 if self.optimizing_criteria=='R2' else Pearson_coeff
        voxel2alpha, alpha2voxel = self.optimize_alpha(data, alpha)
        for alpha, voxels in alpha2voxel.items():
            y_test = np.vstack(Y_test)[:, voxels]
            y_train = np.vstack(Y_train)[:, voxels]
            x_train = np.vstack(X_train)
            self.fit(x_train, y_train, alpha)
            predictions = self.predict(x_test)
            R2[voxels] = self.get_R2_coeff(predictions, y_test)
            Pearson_coeff[voxels] = self.get_Pearson_coeff(predictions, y_test)
        result = {'R2': R2,
                    'Pearson_coeff': Pearson_coeff,
                    'alpha': voxel2alpha
                    }
        return result

    def get_R2_coeff(self, predictions, Y_test):
        """ Compute the R2 score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """
        r2 = r2_score(Y_test, predictions, multioutput='raw_values')
        return r2
    
    def get_Pearson_coeff(self, predictions, Y_test):
        """ Compute the Pearson correlation coefficients
        score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """
        pearson_corr = [pearsonr(Y_test[:,i], predictions[:,i])[0] for i in range(Y_test.shape[1])]
        return pearson_corr
