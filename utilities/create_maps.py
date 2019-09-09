# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import glob

from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
import nibabel as nib

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def check_folder(path):
    # Create adequate folders if necessary
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker


def create_maps(masker, distribution, distribution_name, subject, output_parent_folder, vmax=0.2, pca='', voxel_wise=False):
    model = os.path.basename(output_parent_folder)
    language = os.path.basename(os.path.dirname(output_parent_folder))
    data_type = os.path.basename(os.path.dirname(os.path.dirname(output_parent_folder)))

    img = masker.inverse_transform(distribution)

    pca = 'pca_' + str(pca) if (pca!='') else 'no_pca'
    voxel_wise = 'voxel_wise' if voxel_wise else 'not_voxel_wise'
    
    path2output_raw = os.path.join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.nii.gz')
    path2output_png = os.path.join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.png')

    nib.save(img, path2output_raw)

    display = plot_glass_brain(img, display_mode='lzry', threshold=0, colorbar=True, black_bg=True, vmin=0., vmax=vmax)
    display.savefig(path2output_png)
    display.close()





if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate password from key (and optionnally the login).""")
    parser.add_argument("--output", type=str, default=None, help="Path to output.")
    parser.add_argument("--subject", type=str, default='sub-057', help="Subject name.")
    parser.add_argument("--pca", type=str, default=None, help="Number of components to keep for the PCA.")
    parser.add_argument("--fmri_data", type=str, default='', help="Path to fMRI data directory.")

    args = parser.parse_args()


    fmri_runs = sorted(glob.glob(os.path.join(args.fmri_data, 'fMRI_*run*')))
    masker = compute_global_masker(list(fmri_runs))

    # defining paths
    output_parent_folder = os.path.join(args.output, 'maps')
    alphas = np.mean(np.load(os.path.join(args.output, 'voxel2alpha.npy')), axis=0)
    r2 = np.load(os.path.join(args.output, 'r2.npy'))
    z_values = np.load(os.path.join(args.output, 'z_values.npy'))
    significant_r2 = np.load(os.path.join(args.output, 'r2_significative.npy'))
    pca = int(args.pca) if type(args.pca)==str else None

    check_folder(output_parent_folder)
    
    # creating maps
    create_maps(masker, alphas, 'alphas', args.subject, output_parent_folder, pca=pca, vmax=1000) # alphas # argument deleted: , vmax=5e3
    create_maps(masker, r2, 'r2', args.subject, output_parent_folder, vmax=0.2, pca=pca,  voxel_wise=True) # r2 test
    create_maps(masker, z_values, 'z_values', args.subject, output_parent_folder, pca=pca, voxel_wise=True) # p_values
    create_maps(masker, significant_r2, 'significant_r2', args.subject, output_parent_folder, vmax=0.2, pca=pca, voxel_wise=True) # r2_significative


    