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
import h5py
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

def load_model(path, original_model):
    with h5py.File(path, "r", swmr=True) as fin:
        original_model.coef_ = fin['coef_'][()]
        original_model.intercept_ = fin['intercept_'][()]
        model_parameters = json.loads(fin['parameters'][()])
        new_model = update(original_model, model_parameters)
    return new_model

def save_model(path, model):
    with h5py.File(path, "a", libver='latest') as fout:
        fout.create_dataset('coef_', model.coef_.shape, data=model.coef_)
        fout.create_dataset('intercept_', model.intercept_.shape, data=model.intercept_)
        parameters = vars(model).copy()
        del parameters['coef_']
        del parameters['intercept_']
        fout.create_dataset('parameters', data=json.dumps(parameters))

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
    path2trained_model = os.path.join(args.output, 'ridge_models')
    check_folder(path2trained_model)
    model_loading_path = os.path.join(path2trained_model, 'run_{}_alpha_{}.hdf5'.format(run, alpha))
    model_saving_path = model_loading_path

    try :
        model_fitted = load_model(model_loading_path, model)
    except:
        x_paths = sorted([path[0] for path in [glob.glob(os.path.join(args.x, '*_run{}.npy'.format(i))) for i in indexes]])
        x = [np.load(item) for item in x_paths]

        y_paths = sorted([path[0] for path in [glob.glob(os.path.join(args.y, '*_run{}.npy'.format(i))) for i in indexes]])
        y = [np.load(item) for item in y_paths]
        
        y_train = np.vstack(y)[:, voxels]
        x_train = np.vstack(x)

        model.set_params(alpha=alpha)
        model_fitted = model.fit(x_train, y_train)
        save_model(model_saving_path, model_fitted)
        
    x_test = np.load(glob.glob(os.path.join(args.x, '*_run{}.npy'.format(run)))[0])
    y_test = np.load(glob.glob(os.path.join(args.y, '*_run{}.npy'.format(run)))[0])[:, voxels]
    
    tmp = model_fitted.coef_.copy()

    indexes = args.features_indexes.split(',')
    model_name = args.model_name
    model_fitted.coef_ = np.zeros(model_fitted.coef_.shape)
    model_fitted.coef_[:,int(indexes[0]):int(indexes[1])] = tmp[:,int(indexes[0]):int(indexes[1])]
    r2, pearson_corr = get_score(model_fitted, y_test, x_test)
    
    # sanity check
    path2r2 = os.path.join(args.output, model_name, 'r2')
    path2pearson_corr = os.path.join(args.output, model_name, 'pearson_corr')
    
    check_folder(path2r2)
    check_folder(path2pearson_corr)
    
    # saving
    r2_saving_path = os.path.join(path2r2, 'run_{}_alpha_{}.npy'.format(run, alpha))
    pearson_corr_saving_path = os.path.join(path2pearson_corr, 'run_{}_alpha_{}.npy'.format(run, alpha))
    
    np.save(r2_saving_path, r2)
    np.save(pearson_corr_saving_path, pearson_corr)
    