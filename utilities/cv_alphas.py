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
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--run", type=str, default=None, help="Run split.")
    parser.add_argument("--alphas", type=str, default=None, help="Alphas values to test during CV (delimiter=',').")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")

    args = parser.parse_args()
    #write(checkpoints_path, 'Argparse --> Done')

    #to delete
    checkpoints_path = os.path.join(os.path.dirname(args.input_folder), 'checkpoints_for_run_{}.txt'.format(args.run))

    #write(checkpoints_path, 'Entering variables definition...')

    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    model = Ridge()

    indexes = [int(i) for i in args.indexes.split(',')]
    
    x_paths = sorted([glob.glob(os.path.join(args.x, '*_run{}.npy'.format(i))) for i in indexes]) #[[a], [b], [c]]
    x = [np.load(item[0]) for item in x_paths]

    y_paths = sorted([glob.glob(os.path.join(args.y, '*_run{}.npy'.format(i))) for i in indexes])
    y = [np.load(item[0]) for item in y_paths]

    run = int(args.run)
    nb_voxels = y[0].shape[1]
    nb_alphas = len(alphas)
    nb_runs_cv = len(x)

    #write(checkpoints_path, '\t--> Done')


    logo = LeaveOneOut() # leave on run out !
    cv_index = 0
    logo2 = LeaveOneOut() # leave on run out !
    #write(checkpoints_path, 'Entering main for loop...')
    for train, valid in logo2.split(y):
        #write(checkpoints_path, '\tSplit number: {}'.format(valid))
        y_train = [y[i] for i in train] # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        x_train = [x[i] for i in train]
        dm = np.vstack(x_train)
        fmri = np.vstack(y_train)
        scores = np.zeros((nb_voxels, nb_runs_cv, nb_alphas))
        
        alpha_index = 0
        #write(checkpoints_path, '\t\tData retrieval --> Done')
        #write(checkpoints_path, '\tEntering loop over alphas...')
        for alpha_tmp in alphas: # compute the r2 for a given alpha for all the voxel
            #write(checkpoints_path, '\t\tSetting alpha for model')
            model.set_params(alpha=alpha_tmp)
            ##write(checkpoints_path, '\t\tFitting model of size: {}...'.format(sys.getsizeof(model)))
            model_fitted = model.fit(dm,fmri)
            #write(checkpoints_path, '\t\tComputing r2 scores...')
            r2 = get_r2_score(model_fitted, y[valid[0]], x[valid[0]])
            #write(checkpoints_path, '\t\tConcatenating...')
            scores[:, cv_index, alpha_index] = r2
            alpha_index += 1
            #write(checkpoints_path, '\t\t\t--> Done')
        cv_index += 1
        #write(checkpoints_path, '\t\t--> Done')
    #write(checkpoints_path, '\t--> Done (main for loop)')
    best_alphas_indexes = np.argmax(np.mean(scores, axis=1), axis=1)
    voxel2alpha = np.array([alphas[i] for i in best_alphas_indexes])
    #write(checkpoints_path, 'Voxel2alpha computed. --> size: {}'.format(voxel2alpha.nbytes))

    # compute best alpha for each voxel and group them by alpha-value
    #write(checkpoints_path, 'Computing alpha2voxel...')
    alpha2voxel = {key:[] for key in alphas}
    for index in range(len(voxel2alpha)):
        alpha2voxel[voxel2alpha[index]].append(index)
    #write(checkpoints_path, 'alpha2voxel computed. --> size: {}'.format(alpha2voxel.nbytes))
    #write(checkpoints_path, '\tEntering loop over alphas...')
    for alpha in alphas:
        yaml_path = os.path.join(args.output, 'run_{}_alpha_{}.yml'.format(run, alpha))
        yaml_file = {'alpha': alpha,
                        'voxels': alpha2voxel[alpha],
                        'run': run,
                        'indexes': indexes}

        with open(yaml_path, 'w') as outfile:
            ##write(checkpoints_path, '\tDumping yaml file for alpha={} with size: {} ...'.format(alpha, sys.getsizeof(yaml_file)))
            yaml.dump(yaml_file, outfile, default_flow_style=False)
    #write(checkpoints_path, '\t--> Done (computing best alphas & loop ove ralphas)')
    
    # saving
    #write(checkpoints_path, 'Saving voxel2alpha...')
    np.save(os.path.join(args.output, 'voxel2alpha{}.npy'.format(run)), voxel2alpha)
    #write(checkpoints_path, '\t--> Done')
