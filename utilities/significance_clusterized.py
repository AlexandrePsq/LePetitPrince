# -*- coding: utf-8 -*-

import os
import yaml
import glob
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import Ridge


def get_r2_score(model, y_true, x, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    r2 = r2_score(y_true,
                    model.predict(x),
                    multioutput='raw_values')
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    # return np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])
    return r2

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
        r2_test = get_r2_score(model, y_test, x_test)
        return r2_test, None
    else:
        r2_test = get_r2_score(model, y_test, x_test)
        distribution_array = None
        for index in range(n_sample):
            r2_tmp = get_r2_score(model, y_test, x_test[:, shuffling[index]])
            distribution_array = r2_tmp if distribution_array is None else np.vstack([distribution_array, r2_tmp])
        return r2_test, distribution_array
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--yaml_file", type=str, default=None, help="Path to the yaml file containing alpha, run values and the voxels associateds with.")
    parser.add_argument("--output_r2", type=str, default='', help="Path to the folder of output R2.")
    parser.add_argument("--output_distribution", type=str, default='', help="Path to the folder of output distribution.")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--shuffling", type=str, default='', help="Path to shuffling array.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")
    parser.add_argument("--alpha_percentile", type=str, default=None, help="Value for np.percentile.")

    args = parser.parse_args()

    with open(args.yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    source = 'fMRI'
    model = Ridge()
    alpha = data['alpha']
    voxels = [int(voxel) for voxel in data['voxels']]
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

    # y = fmri[:,voxel].reshape((fmri.shape[0],1))
    model.set_params(alpha=alpha)
    model_fitted = model.fit(x_train, y_train)
    r2, distribution = sample_r2(model_fitted, 
                                    x_test, 
                                    y_test, 
                                    shuffling=np.load(args.shuffling),
                                    n_sample=int(args.n_permutations), 
                                    alpha_percentile=int(args.alpha_percentile))
    # sanity check
    check_folder(args.output_r2)
    check_folder(args.output_distribution)

    # saving
    r2_saving_path = os.path.join(args.output_r2, 'run_{}_voxel_{}.npy'.format(run, voxels))
    distribution_saving_path = os.path.join(args.output_distribution, 'run_{}_voxel_{}.npy'.format(run, voxels))

    np.save(r2_saving_path, r2)
    np.save(distribution_saving_path, distribution)