# -*- coding: utf-8 -*-

import os
import glob
import yaml
import argparse
import sys

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
    parser.add_argument("--nb_runs", type=str, default='', help="Number of runs.")
    parser.add_argument("--run", type=str, default=None, help="Run split.")
    parser.add_argument("--alphas", type=str, default=None, help="Alphas values to test during CV (delimiter=',').")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")
    parser.add_argument("--nb_voxels", type=str, default=None, help="Number of voxels.")
    parser.add_argument("--input", type=str, default=None, help="Path to the folder where the scores of each split are saved.")

    args = parser.parse_args()

    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    nb_alphas = len(alphas)
    indexes = [int(i) for i in args.indexes.split(',')]

    scores = np.zeros((int(args.nb_voxels), int(args.nb_runs), nb_alphas))
    files = sorted(glob.glob(os.path.join(args.input, 'score_run_{}_cv_index_*.npy'.format(args.run))))
    for score in files:
        r2_values = np.load(score)
        cv_index = int(os.path.basename(score).split('_')[5].split('.')[0]) - 1
        scores[:, cv_index, :] = r2_values

    best_alphas_indexes = np.argmax(np.mean(scores, axis=1), axis=1)
    voxel2alpha = np.array([alphas[i] for i in best_alphas_indexes])

    # compute best alpha for each voxel and group them by alpha-value
    alpha2voxel = {key:[] for key in alphas}
    for index in range(len(voxel2alpha)):
        alpha2voxel[voxel2alpha[index]].append(index)
    for alpha in alphas:
        yaml_path = os.path.join(args.output, 'run_{}_alpha_{}.yml'.format(args.run, alpha))
        yaml_file = {'alpha': alpha,
                        'voxels': alpha2voxel[alpha],
                        'run': int(args.run),
                        'indexes': indexes}

        with open(yaml_path, 'w') as outfile:
            yaml.dump(yaml_file, outfile, default_flow_style=False)
    
    # saving
    np.save(os.path.join(args.output, 'voxel2alpha{}.npy'.format(args.run)), voxel2alpha)
