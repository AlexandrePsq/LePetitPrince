# -*- coding: utf-8 -*-

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

def sample_r2(model, x_test, y_test, shuffling, n_sample, alpha_percentile, test=False):
    # receive a trained model, x_test and y_test (test set of the cross-validation).
    # It returns two values (or two lists depending on the parameter voxel_wised):
    # r2 value computed on test set and the distribution array
    if test:
        r2_test, pearson_corr = get_score(model, y_test, x_test)
        return r2_test, pearson_corr, None
    else:
        r2_test, pearson_corr = get_score(model, y_test, x_test)
        distribution_array_r2 = None
        distribution_array_pearson_corr = None
        for index in range(n_sample):
            r2_tmp, pearson_corr_tmp = get_score(model, y_test, x_test[:, shuffling[index]])
            distribution_array_r2 = r2_tmp if distribution_array_r2 is None else np.vstack([distribution_array_r2, r2_tmp])
            distribution_array_pearson_corr = pearson_corr_tmp if distribution_array_pearson_corr is None else np.vstack([distribution_array_pearson_corr, pearson_corr_tmp])
        return r2_test, pearson_corr, distribution_array_r2, distribution_array_pearson_corr
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--yaml_file", type=str, default=None, help="Path to the yaml file containing alpha, run values and the voxels associateds with.")
    parser.add_argument("--output", type=str, default='', help="Path to the folder containing outputs.")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--shuffling", type=str, default='', help="Path to shuffling array.")
    parser.add_argument("--parameters", type=str, default='', help="Path to the yaml file containing the models names and column indexes.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")
    parser.add_argument("--alpha_percentile", type=str, default=None, help="Value for np.percentile.")

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
    
    x_paths = sorted([path[0] for path in [glob.glob(os.path.join(args.x, '*_run{}.npy'.format(i))) for i in indexes]])
    x = [np.load(item) for item in x_paths]

    y_paths = sorted([path[0] for path in [glob.glob(os.path.join(args.y, '*_run{}.npy'.format(i))) for i in indexes]])
    y = [np.load(item) for item in y_paths]
    
    y_train = np.vstack(y)[:, voxels]
    x_train = np.vstack(x)

    x_test = np.load(glob.glob(os.path.join(args.x, '*_run{}.npy'.format(run)))[0])
    y_test = np.load(glob.glob(os.path.join(args.y, '*_run{}.npy'.format(run)))[0])[:, voxels]

    model.set_params(alpha=alpha)
    model_fitted = model.fit(x_train, y_train)
    with open(args.parameters, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print(-1)
            quit()
    tmp = model_fitted.coef_.copy()
    for model in parameters['models']:
        indexes = model['indexes']
        model_name = model['name'] # model['name']=='' if we study the model as a whole
        model_fitted.coef_ = np.zeros(model_fitted.coef_.shape)
        model_fitted.coef_[:,int(indexes[0]):int(indexes[1])] = tmp[:,int(indexes[0]):int(indexes[1])]
        r2, pearson_corr, distribution_array_r2, distribution_array_pearson_corr = sample_r2(model_fitted, 
                                                                                                x_test, 
                                                                                                y_test, 
                                                                                                shuffling=np.load(args.shuffling),
                                                                                                n_sample=int(args.n_permutations), 
                                                                                                alpha_percentile=int(args.alpha_percentile))
        # sanity check
        path2r2 = os.path.join(args.output, model_name, 'r2')
        path2pearson_corr = os.path.join(args.output, model_name, 'pearson_corr')
        path2distribution_array_r2 = os.path.join(args.output, model_name, 'distribution_r2')
        path2distribution_array_pearson_corr = os.path.join(args.output, model_name, 'distribution_pearson_corr')

        check_folder(path2r2)
        check_folder(path2pearson_corr)
        check_folder(path2distribution_array_r2)
        check_folder(path2distribution_array_pearson_corr)

        # saving
        r2_saving_path = os.path.join(path2r2, 'run_{}_alpha_{}.npy'.format(run, alpha))
        pearson_corr_saving_path = os.path.join(path2pearson_corr, 'run_{}_alpha_{}.npy'.format(run, alpha))
        distribution_r2_saving_path = os.path.join(path2distribution_array_r2, 'run_{}_alpha_{}.npy'.format(run, alpha))
        distribution_pearson_corr_saving_path = os.path.join(path2distribution_array_pearson_corr, 'run_{}_alpha_{}.npy'.format(run, alpha))

        np.save(r2_saving_path, r2)
        np.save(pearson_corr_saving_path, pearson_corr)
        np.save(distribution_r2_saving_path, distribution_array_r2)
        np.save(distribution_pearson_corr_saving_path, distribution_array_pearson_corr)