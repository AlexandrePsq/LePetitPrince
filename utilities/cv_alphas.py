# -*- coding: utf-8 -*-

import os
import glob
import yaml
import argparse

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
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    # return np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])
    return r2

def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        check_folder(os.path.dirname(path))
        os.mkdir(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--indexes", type=str, default=None, help="List of run to use for CV (delimiter=',').")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--run", type=str, default=None, help="Run split.")
    parser.add_argument("--alphas", type=str, default=None, help="Alphas values to test during CV (delimiter=',').")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")

    args = parser.parse_args()

    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    model = Ridge()

    indexes = [int(i) for i in args.indexes.split(',')]
    
    x_paths = sorted([glob.glob(os.path.join(args.x, '*_run{}*'.format(i))) for i in indexes])
    x = [np.load(item[0]) for item in x_paths]

    y_paths = sorted([glob.glob(os.path.join(args.y, '*_run{}*'.format(i))) for i in indexes])
    with open(os.path.join(args.output, 'delete_avant.txt'), 'a+') as f:
        f.write(str(y_paths))
        f.write('\n')
    y = [np.load(item[0]) for item in y_paths]

    run = int(args.run)
    nb_voxels = y[0].shape[1]
    nb_alphas = len(alphas)
    nb_runs_cv = len(x)

    


    logo = LeaveOneOut() # leave on run out !
    cv_index = 0
    logo2 = LeaveOneOut() # leave on run out !
    for train, valid in logo2.split(y):
        y_train = [y[i] for i in train] # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        x_train = [x[i] for i in train]
        dm = np.vstack(x_train)
        fmri = np.vstack(y_train)
        scores = np.zeros((nb_voxels, nb_runs_cv, nb_alphas))
        with open(os.path.join(args.output, 'delete_first_part.txt'), 'a+') as f:
            f.write('Et de 1 passage')
            f.write('\n')
        
        alpha_index = 0
        for alpha_tmp in alphas: # compute the r2 for a given alpha for all the voxel
            with open(os.path.join(args.output, 'delete_second_part.txt'), 'a+') as f:
                f.write('Et de 1 passage')
                f.write('\n')
            model.set_params(alpha=alpha_tmp)
            model_fitted = model.fit(dm,fmri)
            r2 = get_r2_score(model_fitted, y[valid[0]], x[valid[0]])
            scores[:, cv_index, alpha_index] = r2
            alpha_index += 1
        cv_index += 1
    best_alphas_indexes = np.argmax(np.mean(scores, axis=1), axis=1)
    voxel2alpha = np.array([alphas[i] for i in best_alphas_indexes])

    # sanity check
    check_folder(args.output)

    # compute best alpha for each voxel and group them by alpha-value
    alpha2voxel = {key:[] for key in alphas}
    for index in range(len(voxel2alpha)):
        alpha2voxel[voxel2alpha[index]].append(index)
    for alpha in alphas:
        yaml_path = os.path.join(args.output, 'run_{}_alpha_{}.yml'.format(run, alpha))
        yaml_file = {'alpha': alpha,
                        'voxels': alpha2voxel[alpha],
                        'run': run,
                        'indexes': indexes}
 
        with open(yaml_path, 'w') as outfile:
            yaml.dump(yaml_file, outfile, default_flow_style=False)
    
    with open(os.path.join(args.output, 'delete_apres.txt'), 'a+') as f:
        f.write('Et de 1 passage')
        f.write('\n')
    
    # saving
    np.save(os.path.join(args.output, 'voxel2alpha{}.npy'.format(run)), voxel2alpha)
