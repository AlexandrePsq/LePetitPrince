# -*- coding: utf-8 -*-
"""
General class for estimator models: fit an array of regressors to fMRI data.
===================================================
A EstimatorModel instanciation requires:
    - model: an instance of sklearn.linear_model that will be use to fit a 
    design-matrix to fMRI data,
    - alpha: int (or None) regularization hyperparameter of the model,
    - alpha_min_log_scale: int minimum of the log scale alpha values that we are testing,
    - alpha_max_log_scale: int maximum of the log scale alpha values that we are testing ,
    - nb_alphas: int, number of alphas to test in our log scale,
    - optimizing_criteria': string specifying the measure to use for optimization (by default
    we use the R2 value).

The mains methods implemented in this class are:
    - self.fit: train the estimator model from {X_train, Y_train, alpha}
    - self.grid_search: compute R2 maps (or other depending on self.optimizing_criteria)
    for multiple values of alphas from models fit on the whole brain.
    - self.optimize_alpha: retrieve the best hyperparameter per voxel from the output
    of the grid_search.
    - self.evaluate: use optimize_alpha to fit a model for each set of voxels having the same 
    hyperparameters and compute the R2/Pearson maps.
"""



import os
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression

from regressors import B2B_reg



class EstimatorModel(object):
    """ General class for estimator models: fit an array
    of regressors to fMRI data.
    """

    def __init__(self, model=Ridge(), alpha=None, alpha_min_log_scale=2, alpha_max_log_scale=4, nb_alphas=25, optimizing_criteria='R2', base=10.0):
        """ Instanciation of EstimatorModel class.
        Arguments:
            - model: sklearn.linear_model
            - alpha: int
            - alpha_min_log_scale: int
            - alpha_max_log_scale: int
            - nb_alphas: int
            - optimizing_criteria: str
            - base: float
        """
        self.alpha = alpha # regularization parameter
        self.model = model
        self.optimizing_criteria = optimizing_criteria
        self.alpha_list = [round(tmp, 5) for tmp in np.logspace(alpha_min_log_scale, alpha_max_log_scale, nb_alphas, base=base)]
    
    def fit(self, X_train, Y_train, alpha):
        """ Fit the model for a given set of runs.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - alpha: float
        """
        if 'Ridge' in str(self.model):
            self.model.set_params(alpha=alpha)
        dm = np.vstack(X_train)
        fmri = np.vstack(Y_train)
        self.model.fit(dm,fmri)
    
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
        """ Fit a model on the whole brain for a list of hyperparameters, 
        and return R2 coefficients, Pearson coefficients and regularization 
        parameters.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - X_test: list (of np.array)
            - Y_test: list (of np.array)
        Returns:
            - result: dict 
                - R2: np.array (2D)
                - Pearson_coeff: np.array (2D)
                - alpha: np.array (1D)
        """
        result = {'R2': [],
                    'Pearson_coeff': [],
                    'alpha': self.alpha_list if ('Ridge' in str(self.model)) else [1]
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
        
    def optimize_alpha(self, data, hyperparameter, nb_voxels=None):
        """ Optimize the hyperparameter of a model given a
        list of measures.
        Arguments:
            - data: np.array (3D)
            - hyperparameter: np.array (2D)
            - nb_voxels: int
        Returns:
            - voxel2alpha: list (of int)
            - alpha2voxel: dict (of list)
        """
        if data is None:
            voxel2alpha = [self.alpha for i in range(nb_voxels)]
        else:
            best_alphas_indexes = np.argmax(np.mean(data, axis=0), axis=0)
            voxel2alpha = np.array([hyperparameter[i] for i in best_alphas_indexes])
        alpha2voxel = {key:[] for key in hyperparameter}
        for index in range(len(voxel2alpha)):
            alpha2voxel[voxel2alpha[index]].append(index)
        return voxel2alpha, alpha2voxel
    
    def evaluate(self, X_train, X_test, Y_train, Y_test, R2=None, Pearson_coeff=None, alpha=None):
        """ Fit a model for each voxel given the parameter optimizing a measure.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - X_test: list (of np.array)
            - Y_test: list (of np.array)
            - R2: np.array (3D)
            - Pearson_coeff: np.array (3D)
            - alpha: np.array (2D)
        The extra dimension of the last 3 arguments results from the ’aggregate_cv’
        method that was applied to the output of ’grid_search’, concatenating cv
        results over a new dimension placed at the index 0.
        Returns:
            - result: dict
        """
        data = R2 if self.optimizing_criteria=='R2' else Pearson_coeff
        alpha = np.mean(alpha, axis=0) if (alpha is not None) else [self.alpha]
        y_test = np.vstack(Y_test)
        x_test = np.vstack(X_test)
        y_train = np.vstack(Y_train)
        x_train = np.vstack(X_train)
        if 'B2B' in str(self.model):
            mean_per_alpha = np.mean(np.mean(data, axis=0), axis=-1)
            alpha_ = alpha[np.argmax(mean_per_alpha)]
            self.fit(x_train, y_train, alpha_)
            predictions = self.predict(x_test)
            R2_ = self.get_R2_coeff(predictions, y_test)
            Pearson_coeff_ = self.get_Pearson_coeff(predictions, y_test)
            coefs = self.model.coef_
            intercepts = self.model.intercept_
            diag_matrix = self.model.diag_matrix_
            voxel2alpha = np.array([alpha_ for i in range(y_train.shape[-1])])
        else:
            R2_ = np.zeros((Y_test[0].shape[1]))
            Pearson_coeff_ = np.zeros((Y_test[0].shape[1]))
            coefs = np.zeros((Y_test[0].shape[1], X_test[0].shape[1]))
            intercepts = np.zeros((Y_test[0].shape[1]))
            diag_matrix = np.array([])

            voxel2alpha, alpha2voxel = self.optimize_alpha(data, alpha, nb_voxels=y_test.shape[-1])
            for alpha_, voxels in alpha2voxel.items():
                if voxels:
                    y_test_ = y_test[:, voxels]
                    y_train_ = y_train[:, voxels]
                    self.fit(x_train, y_train_, alpha_)
                    predictions = self.predict(x_test)
                    R2_[voxels] = self.get_R2_coeff(predictions, y_test_)
                    Pearson_coeff_[voxels] = self.get_Pearson_coeff(predictions, y_test_)
                    coefs[voxels, :] = self.model.coef_
                    intercepts[voxels] = self.model.intercept_
        result = {'R2': R2_,
                    'Pearson_coeff': Pearson_coeff_,
                    'alpha': voxel2alpha,
                    'coef_': coefs,
                    'intercept_': intercepts,
                  'diag_matrix': diag_matrix,
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
        pearson_corr = np.array([pearsonr(Y_test[:,i], predictions[:,i])[0] for i in range(Y_test.shape[1])])
        return pearson_corr
