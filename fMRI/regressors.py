# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class B2B_reg():
    """Implement Bask-to-back regression between X and Y.
    Args:
        - X: np.array
        - Y: np.array
        - m: int
    Returns:
        -
    ref: https://www.biorxiv.org/content/10.1101/2020.03.05.976936v1.full.pdf
    """
    
    def __init__(self, nb_iteration=10, alpha=1.0, linear_model='ridge'):
        """
        """
        self.m = nb_iteration
        self.coef_ = None
        self.intercept_ = None
        self.diag_matrix_ = None
        self.linear_model = linear_model
        self.params = {}
        self.alpha = alpha
        self.name = 'B2B'
        
    def set_params(self, **kwargs):
        self.params = kwargs
        pass
    
    def __name__(self):
        return self.name
    
    def create_reg(self):
        if self.linear_model=='linear_regression':
            return LinearRegression()
        elif self.linear_model=='ridge':
            model = Ridge(self.alpha)
            model.set_params(**self.params)
            return model
    
    def fit(self, X, Y):
        n, dx = X.shape
        _, dy = Y.shape
        E = np.zeros((dx, dx))
        for i in range(self.m):
            np.random.shuffle(X)
            np.random.shuffle(Y)
            X1, X2 = X[:n//2, :], X[n//2:, :]
            Y1, Y2 = Y[:n//2, :], Y[n//2:, :]
            reg_backward = self.create_reg().fit(Y1, X1)
            G = reg_backward.coef_.T
            reg_forward = self.create_reg().fit(X2, np.matmul(Y2, G))
            H = reg_forward.coef_.T
            E[np.diag_indices_from(E)] += np.diag(H)
        E = E / self.m
        reg = self.create_reg().fit(np.matmul(X, E), Y)
        W = reg.coef_
        self.coef_ = W
        self.diag_matrix_ = E
        self.intercept_ = reg.intercept_
    
    def predict(self, X):
        Y = np.matmul(np.matmul(X, self.diag_matrix_), self.coef_.T) + self.intercept_
        return Y
    