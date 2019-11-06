#!/usr/bin/env python3

import os
import yaml
import glob
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
from sklearn.linear_model import Ridge
# import h5py
import json


def get_score(model, y_true, x, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    prediction = model.predict(x)
    r2 = r2_score(y_true,
                    prediction,
                    multioutput='raw_values')
    pearson_corr = [pearsonr(y_true[:,i], prediction[:,i])[0] for i in range(y_true.shape[1])]
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    # return np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])
    return r2, pearson_corr

def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        check_folder(os.path.dirname(path))
        os.mkdir(path)

def update(model, parameters):
    model.alpha = parameters['alpha']
    model.fit_intercept = parameters['fit_intercept']
    model.normalize = parameters['normalize']
    model.copy_X = parameters['copy_X']
    model.max_iter = parameters['max_iter']
    model.tol = parameters['tol']
    model.solver = parameters['solver']
    model.random_state = parameters['random_state']
    model.n_iter_ = parameters['n_iter_']
    return model

def load_model(path2folder, original_model):
    coef_path = os.path.join(path2folder, 'coef_.npy')
    intercept_path = os.path.join(path2folder, 'intercept_.npy')
    parameters_path = os.path.join(path2folder, 'parameters.json')

    original_model.coef_ = np.load(coef_path)
    original_model.intercept_ = np.load(intercept_path)
    with open(parameters_path, 'r') as outfile:
        parameters = json.load(outfile)
    new_model = update(original_model, parameters)
    return new_model

def sample_r2(model, x_test, y_test, shuffling, n_sample):
    # receive a trained model, x_test and y_test (test set of the cross-validation).
    # It returns two values (or two lists depending on the parameter voxel_wised):
    # r2 value computed on test set and the distribution array
    distribution_array_r2 = None
    distribution_array_pearson_corr = None
    for index in range(n_sample):
        r2_tmp, pearson_corr_tmp = get_score(model, y_test, x_test[:, shuffling[index]])
        distribution_array_r2 = r2_tmp if distribution_array_r2 is None else np.vstack([distribution_array_r2, r2_tmp])
        distribution_array_pearson_corr = pearson_corr_tmp if distribution_array_pearson_corr is None else np.vstack([distribution_array_pearson_corr, pearson_corr_tmp])
    return distribution_array_r2, distribution_array_pearson_corr

def write(path, text):
    with open(path, 'a+') as f:
        f.write(text)
        f.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--yaml_file", type=str, default=None, help="Path to the yaml file containing alpha, run values and the voxels associateds with.")
    parser.add_argument("--output", type=str, default='', help="Path to the folder containing outputs.")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--shuffling", type=str, default='', help="Path to shuffling array.")
    parser.add_argument("--n_sample", type=str, default=None, help="Number of permutations.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model.")
    parser.add_argument("--features_indexes", type=str, default=None, help="Indexes of the features to take into account.")

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except :
            print(-1)
            quit()
    
    if data['voxels']==[]:
        quit()

    source = 'fMRI'
    model = Ridge()
    alpha = data['alpha']
    voxels = data['voxels']
    run = int(data['run'])
    indexes = data['indexes']
    model_loading_path = os.path.join(args.output, 'ridge_models', 'run_{}_alpha_{}'.format(run, alpha))
    check_folder(model_loading_path)

    model_fitted = load_model(model_loading_path, model)
    
    x_test = np.load(glob.glob(os.path.join(args.x, '*_run{}.npy'.format(run)))[0])
    y_test = np.load(glob.glob(os.path.join(args.y, '*_run{}.npy'.format(run)))[0])[:, voxels]
    
    tmp = model_fitted.coef_.copy()

    indexes = args.features_indexes.split(',')
    model_name = args.model_name
    model_fitted.coef_ = np.zeros(model_fitted.coef_.shape)
    model_fitted.coef_[:,int(indexes[0]):int(indexes[1])] = tmp[:,int(indexes[0]):int(indexes[1])]
    distribution_array_r2, distribution_array_pearson_corr = sample_r2(model_fitted, 
                                                                        x_test, 
                                                                        y_test, 
                                                                        shuffling=np.load(args.shuffling),
                                                                        n_sample=int(args.n_sample))
    # sanity check
    path2distribution_array_r2 = os.path.join(args.output, model_name, 'distribution_r2')
    path2distribution_array_pearson_corr = os.path.join(args.output, model_name, 'distribution_pearson_corr')
    
    check_folder(path2distribution_array_r2)
    check_folder(path2distribution_array_pearson_corr)

    # saving
    distribution_r2_saving_path = os.path.join(path2distribution_array_r2, 'run_{}_alpha_{}.npy'.format(run, alpha))
    distribution_pearson_corr_saving_path = os.path.join(path2distribution_array_pearson_corr, 'run_{}_alpha_{}.npy'.format(run, alpha))

    np.save(distribution_r2_saving_path, distribution_array_r2)
    np.save(distribution_pearson_corr_saving_path, distribution_array_pearson_corr)




