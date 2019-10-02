# -*- coding: utf-8 -*-

import os
import glob
import yaml
import argparse
import sys
import psutil

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut


def get_r2_score(model, y_true, x, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    r2 = r2_score(y_true,
                    model.predict(x),
                    multioutput='raw_values')
    return r2


def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        check_folder(os.path.dirname(path))
        os.mkdir(path)

def write(path, text):
    with open(path, 'a+') as f:
        f.write(text)
        f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--indexes", type=str, default=None, help="List of run to use for CV (delimiter=',').")
    parser.add_argument("--train", type=str, default=None, help="List of run to use for CV training (delimiter=',').")
    parser.add_argument("--valid", type=str, default=None, help="List of run to use for CV validation (delimiter=',').")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--run", type=str, default=None, help="Run split.")
    parser.add_argument("--alphas", type=str, default=None, help="Alphas values to test during CV (delimiter=',').")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")
    parser.add_argument("--nb_voxels", type=str, default=None, help="Number of voxels.")
    parser.add_argument("--cv_index", type=str, default=None, help="Split index for the CV on alpha")

    args = parser.parse_args()

    checkpoints_path = os.path.join('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/sub-057', 'checkpoints_for_run_{}_valid_{}.txt'.format(args.run, args.valid))
    write(checkpoints_path, 'Entering variables definition...')
    write(checkpoints_path, '{}'.format(vars(args)))

    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    model = Ridge()

    indexes = [int(i) for i in args.indexes.split(',')]
    train = [int(i) for i in args.train.split(',')]
    valid = [int(i) for i in args.valid.split(',')]

    write(checkpoints_path, '\tDefining x and y...')
    write(checkpoints_path, '\tx_paths={}'.format(sorted([glob.glob(os.path.join(args.x, '*_run{}.npy'.format(indexes[i]))) for i in train])))
    
    x_paths = sorted([glob.glob(os.path.join(args.x, '*_run{}.npy'.format(indexes[i]))) for i in train]) #[[a], [b], [c]]
    write(checkpoints_path, '\t\tx_train  ...')
    x_train = [np.load(item[0]) for item in x_paths]
    write(checkpoints_path, '\t\tx_valid ...')
    write(checkpoints_path, '\tx_valid={}'.format(len([np.load(item[0]) for item in x_paths])))
    x_valid = [np.load(item[0]) for item in sorted([glob.glob(os.path.join(args.x, '*_run{}.npy'.format(indexes[i]))) for i in valid])][0]

    y_paths = sorted([glob.glob(os.path.join(args.y, '*_run{}.npy'.format(indexes[i]))) for i in train])
    write(checkpoints_path, '\t\ty_train  ...')
    y_train = [np.load(item[0]) for item in y_paths]
    write(checkpoints_path, '\t\ty_valid ...')
    y_valid = [np.load(item[0]) for item in sorted([glob.glob(os.path.join(args.y, '*_run{}.npy'.format(indexes[i]))) for i in valid])][0]

    write(checkpoints_path, '\tother variables...')
    run = int(args.run)
    nb_voxels = y_train[0].shape[1]
    nb_alphas = len(alphas)
    nb_runs_cv = len(train) + len(valid)
    write(checkpoints_path, 'run: {}\nnb_voxels:{}\nnb_alphas:{}\nnb_runs_cv:{}...'.format(run, nb_voxels, nb_alphas, nb_runs_cv))

    write(checkpoints_path, 'defining dm, fmri, scores...')
    dm = np.vstack(x_train)
    fmri = np.vstack(y_train)
    scores = np.zeros((nb_voxels, 1, nb_alphas))
    
    alpha_index = 0
    write(checkpoints_path, 'entering loop...')
    for alpha_tmp in alphas: # compute the r2 for a given alpha for all the voxel
        model.set_params(alpha=alpha_tmp)
        model_fitted = model.fit(dm,fmri)
        r2 = get_r2_score(model_fitted, y_valid, x_valid)
        scores[:, 0, alpha_index] = r2
        alpha_index += 1
        write(checkpoints_path, '\tloop iteration...')

    saving_path = os.path.join(args.output, 'score_run_{}_cv_index_{}.npy'.format(run, args.cv_index))
    np.save(saving_path, scores)

