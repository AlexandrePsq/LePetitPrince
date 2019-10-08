# -*- coding: utf-8 -*-


import argparse
import os
import numpy as np
import yaml
import glob
import sys
from scipy import stats


def check_folder(path):
    # Create adequate folders if necessary
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def write(path, text):
    with open(path, 'a+') as f:
        f.write(text)
        f.write('\n')


def get_significativity_value(r2_test_array, pearson_corr_array, distribution_r2_array, distribution_pearson_corr_array, alpha_percentile, test=False):
    r2_final = np.mean(r2_test_array, axis=0)
    r2_final = np.array([x if np.abs(x) < 1 else np.sign(x)*0.2 for x in r2_final])
    corr_final = np.mean(pearson_corr_array, axis=0)

    distribution_r2_array_tmp = np.mean(distribution_r2_array, axis=0)
    distribution_pearson_corr_array_tmp = np.mean(distribution_pearson_corr_array, axis=0)

    thresholds_r2 = np.percentile(distribution_r2_array_tmp, alpha_percentile, axis=0) # list: 1 value for each voxel
    thresholds_pearson_corr = np.percentile(distribution_pearson_corr_array_tmp, alpha_percentile, axis=0) # list: 1 value for each voxel
    
    p_values_r2 = (1.0 * np.sum(distribution_r2_array_tmp>r2_final, axis=0))/distribution_r2_array_tmp.shape[0] 
    p_values_pearson_corr =  (1.0 * np.sum(distribution_pearson_corr_array_tmp>corr_final, axis=0))/distribution_pearson_corr_array_tmp.shape[0]

    z_values_r2 = np.array([x if np.abs(x) != np.inf else np.sign(x)*10 for x in np.apply_along_axis(lambda x: stats.norm.ppf(1-x, loc=0, scale=1), 0, p_values_r2)])
    z_values_pearson_corr = np.array([x if np.abs(x) != np.inf else np.sign(x)*10 for x in np.apply_along_axis(lambda x: stats.norm.ppf(1-x, loc=0, scale=1), 0, p_values_pearson_corr)])

    mask_r2 = (r2_final > thresholds_r2)
    mask_pearson_corr = (corr_final > thresholds_pearson_corr)

    mask_pvalues_r2 = (p_values_r2 < 1-alpha_percentile/100)
    mask_pvalues_pearson_corr = (p_values_pearson_corr < 1-alpha_percentile/100)

    return r2_final, corr_final, mask_r2, mask_pearson_corr, mask_pvalues_r2, mask_pvalues_pearson_corr, thresholds_pearson_corr, thresholds_r2, p_values_r2, p_values_pearson_corr, z_values_r2, z_values_pearson_corr



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate password from key (and optionnally the login).""")
    parser.add_argument("--input_folder", type=str, default=None, help="Path to the inputs folder.")
    parser.add_argument("--yaml_files", type=str, default=None, help="Yaml files folder path.")
    parser.add_argument("--nb_runs", type=str, default=None, help="Number of runs.")
    parser.add_argument("--nb_voxels", type=str, default=None, help="Number of voxels.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model.")
    parser.add_argument("--alpha_percentile", type=str, default=None, help="Path of shuffling array.")

    args = parser.parse_args()

    # Variables
    nb_voxels = int(args.nb_voxels)
    nb_runs = int(args.nb_runs)
    scores = np.zeros((nb_runs, nb_voxels))
    corr = np.zeros((nb_runs, nb_voxels))
    distribution_r2_array = np.zeros((nb_runs, int(args.n_permutations), nb_voxels))
    distribution_pearson_corr_array = np.zeros((nb_runs, int(args.n_permutations), nb_voxels))

        
    model_name = args.model_name # model['name']=='' if we study the model as a whole

    input_r2 = os.path.join(args.input_folder, model_name, 'r2')
    input_pearson_corr = os.path.join(args.input_folder, model_name, 'pearson_corr')
    input_distribution_r2 = os.path.join(args.input_folder, model_name, 'distribution_r2')
    input_distribution_pearson_corr = os.path.join(args.input_folder, model_name, 'distribution_pearson_corr')
    
    # retrieving data
    files_r2 = sorted(glob.glob(os.path.join(input_r2, 'run_*_alpha_*.npy'))) # e.g. run_6_alpha_10.32.npy
    files_pearson_corr = sorted(glob.glob(os.path.join(input_pearson_corr, 'run_*_alpha_*.npy'))) # e.g. run_6_alpha_10.32.npy
    files_distribution_r2 = sorted(glob.glob(os.path.join(input_distribution_r2, 'run_*_alpha_*.npy')))
    files_distribution_pearson_corr = sorted(glob.glob(os.path.join(input_distribution_pearson_corr, 'run_*_alpha_*.npy')))

    # merging
    for index in range(len(files_r2)):
        info = os.path.basename(files_r2[index]).split('_')
        run = int(info[1])
        alpha = float(info[3][:-4])
        with open(os.path.join(args.yaml_files, 'run_{}_alpha_{}.yml'.format(run, alpha)), 'r') as stream:
            try:
                voxels = yaml.safe_load(stream)['voxels']
            except :
                print(-1)
                break
        scores[run-1, voxels] = np.load(files_r2[index])
        corr[run-1, voxels] = np.load(files_pearson_corr[index])
        distribution_r2_array[run-1, :, voxels] = np.load(files_distribution_r2[index]).T
        distribution_pearson_corr_array[run-1, :, voxels] = np.load(files_distribution_pearson_corr[index]).T
    alphas_array = []
    for run in range(1, 1+nb_runs):
        alphas_array.append(np.load(os.path.join(args.yaml_files, 'voxel2alpha{}.npy'.format(run))))
    alphas_array = np.vstack(alphas_array)
    
    # computing significativity
    r2_final, corr_final, mask_r2, mask_pearson_corr, mask_pvalues_r2, mask_pvalues_pearson_corr, thresholds_pearson_corr, thresholds_r2, p_values_r2, p_values_pearson_corr, z_values_r2, z_values_pearson_corr = get_significativity_value(scores, 
                                                                                                                                                                                                                                                corr,
                                                                                                                                                                                                                                                distribution_r2_array,
                                                                                                                                                                                                                                                distribution_pearson_corr_array, 
                                                                                                                                                                                                                                                float(args.alpha_percentile))
    r2_significant_with_threshold = np.zeros(r2_final.shape)
    r2_significant_with_threshold[mask_r2] = r2_final[mask_r2]
    r2_significant_with_threshold[~mask_r2] = np.nan

    r2_significant_with_pvalues = np.zeros(r2_final.shape)
    r2_significant_with_pvalues[mask_pvalues_r2] = r2_final[mask_pvalues_r2]
    r2_significant_with_pvalues[~mask_pvalues_r2] = np.nan

    pearson_corr_significant_with_threshold = np.zeros(corr_final.shape)
    pearson_corr_significant_with_threshold[mask_pearson_corr] = corr_final[mask_pearson_corr]
    pearson_corr_significant_with_threshold[~mask_pearson_corr] = np.nan

    pearson_corr_significant_with_pvalues = np.zeros(corr_final.shape)
    pearson_corr_significant_with_pvalues[mask_pvalues_pearson_corr] = corr_final[mask_pvalues_pearson_corr]
    pearson_corr_significant_with_pvalues[~mask_pvalues_pearson_corr] = np.nan

    # defining paths
    output_folder = os.path.join(args.input_folder, model_name, 'outputs')
    check_folder(output_folder)
    r2_output_path = os.path.join(output_folder, 'r2.npy')
    r2_significant_with_threshold_output_path = os.path.join(output_folder, 'r2_significant_with_threshold.npy')
    r2_significant_with_pvalues_output_path = os.path.join(output_folder, 'r2_significant_with_pvalues.npy')

    pearson_corr_output_path = os.path.join(output_folder, 'pearson_corr.npy')
    pearson_corr_significant_with_threshold_output_path = os.path.join(output_folder, 'pearson_corr_significant_with_threshold.npy')
    pearson_corr_significant_with_pvalues_output_path = os.path.join(output_folder, 'pearson_corr_significant_with_pvalues.npy')

    mask_r2_with_threshold_output_path = os.path.join(output_folder, 'mask_r2_with_threshold.npy')
    mask_r2_with_pvalues_output_path = os.path.join(output_folder, 'mask_r2_with_pvalues.npy')
    mask_pearson_corr_with_threshold_output_path = os.path.join(output_folder, 'mask_pearson_corr_with_threshold.npy')
    mask_pearson_corr_with_pvalues_output_path = os.path.join(output_folder, 'mask_pearson_corr_with_pvalues.npy')

    thresholds_r2_output_path = os.path.join(output_folder, 'thresholds_r2.npy')
    thresholds_pearson_corr_output_path = os.path.join(output_folder, 'thresholds_pearson_corr.npy')

    z_values_r2_output_path = os.path.join(output_folder, 'z_values_r2.npy')
    z_values_pearson_corr_output_path = os.path.join(output_folder, 'z_values_pearson_corr.npy')

    p_values_r2_output_path = os.path.join(output_folder, 'p_values_r2.npy')
    p_values_pearson_corr_output_path = os.path.join(output_folder, 'p_values_pearson_corr.npy')

    distribution_r2_output_path = os.path.join(output_folder, 'distribution_r2.npy')
    distribution_pearson_corr_output_path = os.path.join(output_folder, 'distribution_pearson_corr.npy')

    alphas_path = os.path.join(output_folder, 'voxel2alpha.npy')

    # saving
    np.save(r2_output_path, r2_final)
    np.save(r2_significant_with_threshold_output_path, r2_significant_with_threshold)
    np.save(r2_significant_with_pvalues_output_path, r2_significant_with_pvalues)

    np.save(pearson_corr_output_path, corr_final)
    np.save(pearson_corr_significant_with_threshold_output_path, pearson_corr_significant_with_threshold)
    np.save(pearson_corr_significant_with_pvalues_output_path, pearson_corr_significant_with_pvalues)

    np.save(mask_r2_with_threshold_output_path, mask_r2)
    np.save(mask_r2_with_pvalues_output_path, mask_pvalues_r2)
    np.save(mask_pearson_corr_with_threshold_output_path, mask_pearson_corr)
    np.save(mask_pearson_corr_with_pvalues_output_path, mask_pvalues_pearson_corr)

    np.save(thresholds_r2_output_path, thresholds_r2)
    np.save(thresholds_pearson_corr_output_path, thresholds_pearson_corr)

    np.save(z_values_r2_output_path, z_values_r2)
    np.save(z_values_pearson_corr_output_path, z_values_pearson_corr)

    np.save(p_values_r2_output_path, p_values_r2)
    np.save(p_values_pearson_corr_output_path, p_values_pearson_corr)

    np.save(distribution_r2_output_path, distribution_r2_array)
    np.save(distribution_pearson_corr_output_path, distribution_pearson_corr_array)

    np.save(alphas_path, alphas_array)
