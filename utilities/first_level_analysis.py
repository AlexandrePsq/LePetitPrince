from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
import nibabel as nib
from .splitter import Splitter
import sklearn
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from .utils import get_scores, get_significativity_value, generate_random_prediction, get_output_parent_folder
from .settings import Params, Paths
import os
import pickle
from os.path import join
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from time import time

import warnings
warnings.simplefilter(action='ignore')


params = Params()
paths = Paths()

#########################################
################ Figures ################
#########################################

def create_maps(masker, distribution, distribution_name, subject, output_parent_folder, vmax=None, pca='', voxel_wise=False):
    """Compute maps from numpy vector. Plots glass brain representation using a masker extracted from fMRI data.
    :masker: (MultiNiftiMasker) function that can project a 3D representation to 1D and 1D to 3D.
    :distribution: (numpy array) distribution to plot.
    :distribution_name: (str) distribution name.
    :subject: (str) name of the subject (e.g.: sub-xxx).
    :output_parent_folder: (str) path to the folder containing the outputs of the function.
    :vmax: maximum value on the colored bar of the plot.
    :pca: (str) number of components used for the PCA ('' if no PCA).
    :voxel_wise: (bool) precise if we specified a model per voxel.
    """
    model = os.path.basename(output_parent_folder)
    language = os.path.basename(os.path.dirname(output_parent_folder))
    data_type = os.path.basename(os.path.dirname(os.path.dirname(output_parent_folder)))

    img = masker.inverse_transform(distribution)

    pca = 'pca_' + str(pca) if params.pca else 'no_pca'
    voxel_wise = 'voxel_wise' if voxel_wise else 'not_voxel_wise'
    
    path2output_raw = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.nii.gz')
    path2output_png = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.png')

    nib.save(img, path2output_raw)

    display = plot_glass_brain(img, display_mode='lzry', threshold=0, colorbar=True, black_bg=True, vmin=0., vmax=vmax)
    display.savefig(path2output_png)
    display.close()



#########################################
######### First level analysis ##########
#########################################


def compute_global_masker(files):
    """Return a MultiNiftiMasker object.
    :files: (list) (e.g.: [[path, path2], [path3, path4]])
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=params.pref.detrend, standardize=params.pref.standardize) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker


def do_single_subject(subject, fmri_filenames, matrices, masker, source, data_type, language, model, voxel_wised=False, alpha_list=np.logspace(-3, -1, 30), pca=params.n_components_default):
    """Main function for computing r2 and pearson maps for a subject for a given set of design-matrices (associated with one model)
    and creating the associated maps.
    :subject : (str) Name of the subject (e.g.: sub-xxx).
    :fmri_filenames: (list) Paths of the fMRI data runs (1 for each run).
    :matrices: (list) list of numpy array design matrices (1 for each run).
    :masker: (MultiNiftiMasker object) function that can project a 3D representation to 1D and 1D to 3D.
    :source: (str - optional) Source of acquisition used (fMRI or MEG).
    :language : (str) language used.
    :data_type: (str) Type of data (fMRI, MEG, text, wave) or step of the pipeline 
    (raw-features, features, design-matrices, glm-indiv, ridge-indiv, etc...).
    :model: (sklearn.linear_model) Model trained on a set of voxels.
    :voxel_wised: (bool) Precise if we fit a model per voxel or for the whole brain.
    :alpha_list: (np.array) List of alpha to test for voxel-wised model.
    :pca: (int) Number of components to keep if a pca was applied.
    """
    fmri_runs = [masker.transform(f) for f in fmri_filenames] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column
    output_parent_folder = get_output_parent_folder(source, data_type, language, model)
    
    if voxel_wised:
        alphas, r2, pearson_corr, distribution_array_r2, distribution_array_pearson = per_voxel_analysis(model, fmri_runs, matrices, subject, alpha_list)
        alphas = np.mean(alphas, axis=0)
        create_maps(masker, alphas, 'alphas', subject, output_parent_folder, pca=pca)
    else:
        r2, pearson_corr, distribution_array_r2, distribution_array_pearson = whole_brain_analysis(model, fmri_runs, matrices, subject)
    optional = '_' + str(params.pref.alpha_default) if ((type(model) == sklearn.linear_model.Ridge) & (not params.voxel_wise)) else ''
    r2 = np.mean(r2, axis=0)
    pearson_corr = np.mean(pearson_corr, axis=0)
    distribution_array_r2 = np.mean(distribution_array_r2, axis=0)
    distribution_array_pearson = np.mean(distribution_array_pearson, axis=0)
    # Compute r2 and pearson maps and save them under .nii.gz and .png formats
    create_maps(masker, r2, 'r2{}'.format(optional), subject, output_parent_folder, vmax=0.2, pca=pca,  voxel_wise=voxel_wised) 
    create_maps(masker, pearson_corr, 'pearson{}'.format(optional), subject, output_parent_folder, vmax=0.55, pca=pca,  voxel_wise=voxel_wised) 

    # Compute significant values
    try:
        r2_significative, mask_r, z_values_r, p_values_r = get_significativity_value(r2, 
                                                                                        distribution_array_r2, 
                                                                                        params.alpha_percentile)
        pearson_significative, mask_p, z_values_p, p_values_p = get_significativity_value(pearson_corr, 
                                                                                            distribution_array_pearson, 
                                                                                            params.alpha_percentile)
        path2mask_r = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(os.path.basename(os.path.dirname(os.path.dirname(output_parent_folder))), 
                                                                                    os.path.basename(os.path.dirname(output_parent_folder)), 
                                                                                    os.path.basename(output_parent_folder), 
                                                                                    'mask_r2', 
                                                                                    'pca_' + str(pca) if params.pca else 'no_pca',
                                                                                    'voxel_wise' if voxel_wised else 'not_voxel_wise',
                                                                                    subject)
                                                                                    +'.npy')
        path2mask_p = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(os.path.basename(os.path.dirname(os.path.dirname(output_parent_folder))), 
                                                                                    os.path.basename(os.path.dirname(output_parent_folder)), 
                                                                                    os.path.basename(output_parent_folder), 
                                                                                    'mask_pearson', 
                                                                                    'pca_' + str(pca) if params.pca else 'no_pca',
                                                                                    'voxel_wise' if voxel_wised else 'not_voxel_wise',
                                                                                    subject)
                                                                                    +'.npy')
        np.save(path2mask_r, mask_r) # save mask r2
        np.save(path2mask_p, mask_p) # save mask pearson
        create_maps(masker, z_values_r, 'z-values-r2{}'.format(optional), subject, output_parent_folder, pca=pca, voxel_wise=voxel_wised) # z_values r2
        create_maps(masker, z_values_p, 'z-values-pearson{}'.format(optional), subject, output_parent_folder, pca=pca, voxel_wise=voxel_wised) # z_values pearson
        create_maps(masker, p_values_r, 'p-values-r2{}'.format(optional), subject, output_parent_folder, pca=pca, voxel_wise=voxel_wised) # p_values r2
        create_maps(masker, p_values_p, 'p-values-pearson{}'.format(optional), subject, output_parent_folder, pca=pca, voxel_wise=voxel_wised) # p_values pearson
        create_maps(masker, r2_significative, 'significative_r2{}'.format(optional), subject, output_parent_folder, vmax=0.30, pca=pca, voxel_wise=voxel_wised) # r2_significative
        create_maps(masker, pearson_significative, 'significative_pearson{}'.format(optional), subject, output_parent_folder, vmax=0.55, pca=pca, voxel_wise=voxel_wised) # r2_significative
    except Exception as e:
        print('R2 and Pearson safely computed.')
        print(e)


def whole_brain_analysis(model, fmri_runs, design_matrices, subject):
    """Compute r2 and pearson coefficient given a list of design-matrices 
    associated with a given model.
    :model: (sklearn.linear_model) Model trained on a set of voxels.
    :fmri_runs: (list of numpy arrays) list of fMRI data runs (1 for each run).
    :design_matrices: (list of numpy arrays) list of design matrices (1 for each run).
    :subject : (str) Name of the subject (e.g.: sub-xxx).
    """
    nb_runs     = len(fmri_runs)
    nb_voxels   = fmri_runs[0].shape[1]
    n_sample    = params.n_sample

    distribution_array_r2       = np.zeros((nb_runs, n_sample, nb_voxels))
    distribution_array_pearson  = np.zeros((nb_runs, n_sample, nb_voxels))
    r2                          = np.zeros((nb_runs, nb_voxels))
    pearson_corr                = np.zeros((nb_runs, nb_voxels))
    
    logo = LeaveOneGroupOut() # leave on run out !
    cv = 0
    for train, test in tqdm(logo.split(fmri_runs, groups=range(1, nb_runs+1))):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train]) # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train = np.vstack([design_matrices[i] for i in train])

        if type(model) == sklearn.linear_model.RidgeCV:
            nb_samples = np.cumsum([0] + [[fmri_runs[i] for i in train][i].shape[0] for i in range(len([fmri_runs[i] for i in train]))]) # list of cumulative lenght
            indexes = {'run{}'.format(run+1): [nb_samples[i], nb_samples[i+1]] for i, run in enumerate(train)}
            model.cv = Splitter(indexes_dict=indexes, n_splits=nb_runs-1) # adequate splitter for cross-validate alpha taking into account groups
        model_fitted = model.fit(predictors_train, fmri_data_train)

        # return the r2 score and pearson correlation coefficient for each voxel (=list)
        r2_split, pearson_corr_split = get_scores(model_fitted, fmri_runs[test[0]], 
                                        design_matrices[test[0]], r2_min=-0.5, 
                                        r2_max=0.99, pearson_min=-0.2, pearson_max=0.99, 
                                        output='both')
        # Compute distribution over the randomly shuffled columns of x_test
        distribution_array_r2_split, distribution_array_pearson_corr_split = generate_random_prediction(model_fitted, 
                                                                                                        design_matrices[test[0]], 
                                                                                                        fmri_runs[test[0]])
        
        # Update
        r2[cv, :]                           = r2_split # r2_split is a 1d array: 1 value for each voxel
        pearson_corr[cv, :]                 = pearson_corr_split # pearson_corr_split is a 1d array: 1 value for each voxel
        distribution_array_r2[cv, :, :]     = distribution_array_r2_split # distribution is a 2d array: n_sample values for each voxel
        distribution_array_pearson[cv, :, :]= distribution_array_pearson_corr_split # distribution is a 2d array: n_sample values for each voxel
        cv += 1
    return r2, pearson_corr, distribution_array_r2, distribution_array_pearson


def per_voxel_analysis(model, fmri_runs, design_matrices, subject, alpha_list):
    """Compute r2 and pearson coefficient given a list of design-matrices 
    associated with a given model, fitting a model per voxel.
    :model: (sklearn.linear_model) Model trained on a set of voxels.
    :fmri_runs: (list of numpy arrays) list of fMRI data runs (1 for each run).
    :design_matrices: (list of numpy arrays) list of design matrices (1 for each run).
    :subject : (str) Name of the subject (e.g.: sub-xxx).
    :alpha_list: (np.array) List of alpha to test for voxel-wised model.
    """
    n_sample = params.n_sample
    nb_voxels = fmri_runs[0].shape[1]
    nb_alphas = len(alpha_list)
    nb_runs = len(fmri_runs)
    nb_runs_valid = nb_runs -1 

    alphas                      = np.zeros((nb_runs, nb_voxels))
    distribution_array_r2       = np.zeros((nb_runs, n_sample, nb_voxels))
    distribution_array_pearson  = np.zeros((nb_runs, n_sample, nb_voxels))
    r2                          = np.zeros((nb_runs, nb_voxels))
    pearson_corr                = np.zeros((nb_runs, nb_voxels))

    # loop for r2 computation
    cv3 = 0
    logo = LeaveOneOut() # leave on run out !
    for train_, test in logo.split(fmri_runs):
        fmri_data_train_ = [fmri_runs[i] for i in train_] # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train_ = [design_matrices[i] for i in train_]
        
        cv2 = 0
        logo2 = LeaveOneOut() # leave on run out !
        for train, valid in logo2.split(fmri_data_train_):
            fmri_data_train = [fmri_data_train_[i] for i in train] # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
            predictors_train = [predictors_train_[i] for i in train]
            dm = np.vstack(predictors_train)
            fmri = np.vstack(fmri_data_train)
            scores_cv1 = np.zeros((nb_voxels, nb_runs_valid, nb_alphas))
            
            cv1 = 0
            for alpha_tmp in tqdm(alpha_list): # compute the r2 for a given alpha for all the voxel
                model.set_params(alpha=alpha_tmp)
                model_fitted = model.fit(dm,fmri)
                r2 = get_scores(model_fitted, fmri_data_train_[valid[0]], predictors_train_[valid[0]], output='r2')
                scores_cv1[:, cv2, cv1] = r2
                cv1 += 1
            cv2 += 1
        best_alphas_indexes = np.argmax(np.mean(scores_cv1, axis=1), axis=1)
        alphas[cv3, :] = np.array([alpha_list[i] for i in best_alphas_indexes])
        fmri2 = np.vstack(fmri_data_train_)
        dm2 = np.vstack(predictors_train_)
        for voxel in tqdm(range(nb_voxels)): # loop through the voxels and fit the model with the best alpha for this voxel
            y = fmri2[:,voxel].reshape((fmri2.shape[0],1))
            model.set_params(alpha=alphas[cv3, voxel])
            model_fitted = model.fit(dm2, y)

            # return the r2 score and pearson correlation coefficient for each voxel
            r2_split, pearson_corr_split = get_scores(model_fitted, 
                                                        fmri_runs[test[0]][:,voxel].reshape((fmri_runs[test[0]].shape[0],1)), 
                                                        design_matrices[test[0]], r2_min=-0.5, 
                                                        r2_max=0.99, pearson_min=-0.2, pearson_max=0.99, 
                                                        output='both')
            # Compute distribution over the randomly shuffled columns of x_test
            distribution_array_r2_split, distribution_array_pearson_split = generate_random_prediction(model_fitted, 
                                                                                                            design_matrices[test[0]], 
                                                                                                            fmri_runs[test[0]][:,voxel].reshape((fmri_runs[test[0]].shape[0],1)))
            r2[cv3, voxel] = r2_split[0]
            pearson_corr[cv3, voxel] = pearson_corr_split[0]
            distribution_array_r2[cv3, :, voxel] = distribution_array_r2_split
            distribution_array_pearson[cv3, :, voxel] = distribution_array_pearson_split

        cv3 += 1
        
    return alphas, r2, pearson_corr, distribution_array_r2, distribution_array_pearson
    