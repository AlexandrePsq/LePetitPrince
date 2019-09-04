# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import glob


def check_folder(path):
    # Create adequate folders if necessary
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def get_significativity_value(r2_test_array, distribution_array, alpha_percentile, test=False):
    # receive r2 computed on the test set and the entire distribution of r2 values computed accross
    # all runs (with shuffled columns) and returns r2 averaged across CV, significative r2 and threshold

    r2_final = np.mean(r2_test_array, axis=0)
    distribution_array = np.mean(distribution_array, axis=0)

    z_values = np.percentile(distribution_array, alpha_percentile, axis=0) # list: 1 value for each voxel
    mask = (r2_final > z_values).astype(int)
    
    return r2_final, mask, z_values, distribution_array



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate password from key (and optionnally the login).""")
    parser.add_argument("--input_r2", type=str, default=None, help="Path to the folder of input R2.")
    parser.add_argument("--input_distribution", type=str, default=None, help="Path to the folder of input distribution.")
    parser.add_argument("--output", type=str, default=None, help="Output folder path.")
    parser.add_argument("--yaml_files", type=str, default=None, help="Yaml files folder path.")
    parser.add_argument("--nb_runs", type=str, default=None, help="Number of runs.")
    parser.add_argument("--nb_voxels", type=str, default=None, help="Number of voxels.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")
    parser.add_argument("--alpha_percentile", type=str, default=None, help="Path of shuffling array.")

    args = parser.parse_args()

    # Variables
    nb_voxels = int(args.nb_voxels)
    nb_runs = int(args.nb_runs)
    scores = np.zeros((nb_runs, nb_voxels))
    distribution_array = np.zeros((nb_runs, int(args.n_permutations), nb_voxels))
    
    # retrieving data
    files_r2 = sorted(glob.glob(os.path.join(args.input_r2, 'run_*_voxel_*.npy'))) # e.g. run_6_voxel_[1,2,3].npy
    files_distribution = sorted(glob.glob(os.path.join(args.input_distribution, 'run_*_voxel_*.npy')))

    # merging
    for index in range(len(files_r2)):
        info = os.path.basename(files_r2[index]).split('_')
        run = int(info[1])
        voxels = eval(info[3].split('.')[0])
        scores[run, voxels] = np.load(files_r2[index])
        distribution_array[run, :, voxels] = np.load(files_distribution[index]).T
    
    alphas_array = []
    for run in range(nb_runs):
        alphas_array.append(np.load(os.path.join(args.yaml_files, 'voxel2alpha{}.npy'.format(run))))
    alphas_array = np.vstack(alphas_array)
    
    # computing significativity
    r2, mask, z_values, distribution_array = get_significativity_value(scores, 
                                                                        distribution_array, 
                                                                        int(args.alpha_percentile))
    r2_significative = np.zeros(r2.shape)
    r2_significative[mask] = r2[mask]
    r2_significative[~mask] = -1

    # defining paths
    check_folder(args.output)
    r2_output_path = os.path.join(args.output, 'r2.npy')
    r2_significative_output_path = os.path.join(args.output, 'r2_significative.npy')
    mask_output_path = os.path.join(args.output, 'mask.npy')
    z_values_output_path = os.path.join(args.output, 'z_values.npy')
    distribution_output_path = os.path.join(args.output, 'distribution.npy')
    alphas_path = os.path.join(args.output, 'voxel2alpha.npy')

    # saving
    np.save(r2_output_path, r2)
    np.save(r2_significative_output_path, r2_significative)
    np.save(mask_output_path, mask)
    np.save(z_values_output_path, z_values)
    np.save(distribution_output_path, distribution_array)
    np.save(alphas_path, alphas_array)
