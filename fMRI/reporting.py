import os
import umap
import gc
import glob
import itertools
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from textwrap import wrap
from joblib import dump, load
from itertools import combinations
from joblib import Parallel, delayed

import matplotlib
import seaborn as sns
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact

import scipy
import hdbscan
import nistats
import nilearn
import nibabel as nib
from nilearn import plotting
from nilearn import datasets
from scipy.stats import norm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA 
from sklearn import manifold
from sklearn.neighbors import kneighbors_graph
from nilearn.regions import RegionExtractor
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from sklearn.cluster import AgglomerativeClustering, KMeans
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_surf_roi
from nilearn.surface import vol_to_surf

from logger import Logger
from linguistics_info import *
from utils import read_yaml, check_folder, fetch_masker, possible_subjects_id, get_subject_name, fetch_data, get_roi_mask, filter_args, load_masker


SEED = 1111


#########################################
############ Basic functions ############
#########################################

def fetch_map(path, distribution_name):
    path = os.path.join(path, '*' + distribution_name+'.nii.gz')
    files = sorted(glob.glob(path))
    return files

def load_atlas(name='cort-maxprob-thr25-2mm', symmetric_split=True):
    atlas = datasets.fetch_atlas_harvard_oxford(name, symmetric_split=symmetric_split)
    labels = atlas['labels']
    maps = nilearn.image.load_img(atlas['maps'])
    return maps, labels

def batchify(x, y, size=10):
    if len(x) != len(y):
        raise ValueError('vector length mismatch')
    m = len(x)
    x_batch = []
    y_batch = []
    last = 0
    for _ in range(m//size):
        x_batch.append(x[last:last+size])
        y_batch.append(y[last:last+size])
        last = last+size
    x_batch.append(x[last:])
    y_batch.append(y[last:])
    return zip(x_batch, y_batch)

def get_possible_hrf():
    hrf_list = [
    'spm', # hrf model used in SPM
    'spm + derivative', # SPM model plus its time derivative (2 regressors)
    'spm + derivative + dispersion', # idem, plus dispersion derivative (3 regressors)
    'glover', # this one corresponds to the Glover hrf
    'glover + derivative', # the Glover hrf + time derivative (2 regressors)
    'glover + derivative + dispersion'] # idem + dispersion derivative
    return hrf_list

def get_default_analysis():
    analysis = {
    'Hidden-layers': 'hidden-layer-*',
    'Attention-layers': 'attention_layer-*',
    'Specific-attention-heads': 'attention-layer-*_head-*'}
    return analysis

def get_layers_data(
    model_names, 
    analysis=None, 
    language='english',
    OUTPUT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english/'
    ):
    data = {}
    if analysis is None:
        analysis = get_default_analysis()
    for model_name in model_names:
        data[model_name] = {}
        for analysis_of_interest in analysis.keys():
            R2_maps = {}
            Pearson_coeff_maps = {}
            data[model_name][analysis_of_interest] = {}
            for subject_id in tqdm(possible_subjects_id(language)):
                subject = get_subject_name(subject_id)
                path_to_map = os.path.join(OUTPUT_PATH, subject, '_'.join([model_name, str(subject_id), analysis[analysis_of_interest]]))
                R2_maps[subject] = fetch_map(path_to_map, 'R2')
                if analysis_of_interest == 'Attention-layers':
                    R2_maps[subject] = [map_ for map_ in R2_maps[subject] if 'head' not in map_]
                Pearson_coeff_maps[subject] = fetch_map(path_to_map, 'Pearson_coeff')
                if analysis_of_interest == 'Attention-layers':
                    Pearson_coeff_maps[subject] = [map_ for map_ in Pearson_coeff_maps[subject] if 'head' not in map_]
                print(subject, '-', len(R2_maps[subject]), '-', len(Pearson_coeff_maps[subject]))
            if analysis_of_interest == 'Hidden-layers':
                data[model_name][analysis_of_interest]['models'] = [os.path.dirname(name).split('_')[-1] for name in list(Pearson_coeff_maps.values())[-1]]
            elif analysis_of_interest == 'Attention-layers':
                if 'bert-' in model_name:
                    data[model_name][analysis_of_interest]['models'] = ['_'.join(os.path.dirname(name).split('_')[-2:]) for name in list(Pearson_coeff_maps.values())[-1]]
                else:
                    data[model_name][analysis_of_interest]['models'] = [os.path.dirname(name).split('_')[-1] for name in list(Pearson_coeff_maps.values())[-1]]
            else:
                data[model_name][analysis_of_interest]['models'] = ['_'.join(os.path.dirname(name).split('_')[-2:]) for name in list(Pearson_coeff_maps.values())[-1]]
            R2_lists = list(zip(*R2_maps.values()))
            Pearson_coeff_lists = list(zip(*Pearson_coeff_maps.values()))
            for index, model in enumerate(data[model_name][analysis_of_interest]['models']):
                try:
                    data[model_name][analysis_of_interest][model] = {'R2': list(R2_lists[index]),
                                                                    'Pearson_coeff': list(Pearson_coeff_lists[index])
                                                                    }
                except:
                    print('ERROR with: ', model_name)
                    print(analysis_of_interest)
                    print(model)
                    print(index)
                    print(data[model_name][analysis_of_interest]['models'])
    return data

def get_model_data(
    model_names, 
    language='english',
    OUTPUT_PATH="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english"
    ):
    data_full = {}
    for model_name in model_names:
        data_full[model_name.replace('_{}', '')] = {}
        R2_maps = {}
        Pearson_coeff_maps = {}
        for subject_id in tqdm(possible_subjects_id(language)):
            subject = get_subject_name(subject_id)
            path_to_map = os.path.join(OUTPUT_PATH, subject, model_name.format(subject_id))
            R2_maps[subject] = fetch_map(path_to_map, 'R2')
            Pearson_coeff_maps[subject] = fetch_map(path_to_map, 'Pearson_coeff')
            print(subject, '-', len(R2_maps[subject]), '-', len(Pearson_coeff_maps[subject]))
        R2_lists = list(zip(*R2_maps.values()))
        Pearson_coeff_lists = list(zip(*Pearson_coeff_maps.values()))
        try:
            data_full[model_name.replace('_{}', '')] = {'R2': list(R2_lists[0]),
                                'Pearson_coeff': list(Pearson_coeff_lists[0])
                                }
        except:
            print('ERROR with: ', model_name)
    return data_full

def check_data(data, N):
    for key in data.keys():
        if 'R2' in data[key]:
            assert len(data[key]['R2'])==N
        if 'Pearson_coeff' in data[key]:
            assert len(data[key]['Pearson_coeff'])==N

def extract_roi_data(map_, masker, threshold_img=None, voxels_filter=None):
    try:
        array = masker.transform(map_)
        if threshold_img is not None:
            threshold = masker.transform(threshold_img)
            if voxels_filter is not None:
                mask = threshold >= voxels_filter
                array = array[mask]
                threshold = threshold[mask]
            threshold[threshold==0.0] = 10**8 # due to some voxel equal to 0 in threshold image we have nan value
            array = np.divide(array, threshold)
        maximum = np.max(array)
        third_quartile = np.percentile(array, 75)
        mean = np.mean(array)
        size = len(array.reshape(-1))
    except ValueError:
        maximum = np.nan
        third_quartile = np.nan
        mean = np.nan
        size = np.nan
    return mean, third_quartile, maximum, size

def fit_per_roi(maps, atlas_maps, labels, global_mask, PROJECT_PATH, threshold_img=None, voxels_filter=None):
    print("\tLooping through labeled masks...")
    mean = np.zeros((len(labels), len(maps)))
    third_quartile = np.zeros((len(labels), len(maps)))
    maximum = np.zeros((len(labels), len(maps)))
    size = np.zeros((len(labels), len(maps)))
    for index_mask in tqdm(range(len(labels))):
        masker = get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask, PROJECT_PATH=PROJECT_PATH)
        results = Parallel(n_jobs=-2)(delayed(extract_roi_data)(
            map_, 
            masker, 
            threshold_img=threshold_img,
            voxels_filter=voxels_filter
        ) for map_ in maps)
        results = list(zip(*results))
        mean[index_mask, :] = np.hstack(np.array(results[0]))
        third_quartile[index_mask, :] = np.hstack(np.array(results[1]))
        maximum[index_mask, :] = np.hstack(np.array(results[2]))
        size[index_mask, :] = np.hstack(np.array(results[3]))
            
            
    print("\t\t-->Done")
    return mean, third_quartile, maximum, size

def get_data_per_roi(
    data, 
    atlas_maps,
    labels,
    global_mask=None,
    analysis=None, 
    model_name=None,
    threshold_img=None, 
    voxels_filter=None,
    language='english', 
    object_of_interest='Pearson_coeff', 
    attention_head_reordering=[0, 1, 2, 3, 6, 4, 5, 7],
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    models = None
    if analysis is None:
        maps = []
        for model_name in data.keys():
            path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, model_name))
            name = '{}_group_fdr_effect'.format(object_of_interest)
            maps.append(fetch_map(path, name)[0])
        plot_name = ["Model comparison"]
        # extract data
        mean, third_quartile, maximum, size = fit_per_roi(maps, atlas_maps, labels, global_mask=global_mask, threshold_img=threshold_img, voxels_filter=voxels_filter, PROJECT_PATH=PROJECT_PATH)
        result = {
                    'maximum': maximum,
                    'third_quartile': third_quartile,
                    'mean': mean,
                    'models': data.keys(),
                    'labels': labels,
                    'plot_name': '',
                    'size': size
                }
    else:
        result = {}
        for key in analysis:
            result[key] = []
            for model_name in data.keys():
                maps = []
                for model in data[model_name][key].keys():
                    if model == 'models':
                        models = np.array(data[model_name][key][model])
                    else:
                        name = '_'.join([model_name, model])
                        path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
                        name = '{}_group_fdr_effect'.format(object_of_interest)
                        maps.append(fetch_map(path, name)[0])
                plot_name = ["{} for {}".format(model_name, key)]

                # extract data
                mean, third_quartile, maximum, size = fit_per_roi(maps, atlas_maps, labels, global_mask=global_mask, threshold_img=threshold_img, voxels_filter=voxels_filter, PROJECT_PATH=PROJECT_PATH)
                print("\t\t-->Done")
                # Small reordering of models so that layers are in increasing order
                if key=='Hidden-layers':
                    mean = np.hstack([mean[:, :2], mean[:,5:], mean[:,2:5]])
                    third_quartile = np.hstack([third_quartile[:, :2], third_quartile[:,5:], third_quartile[:,2:5]])
                    maximum = np.hstack([maximum[:, :2], maximum[:,5:], maximum[:,2:5]])
                    size = np.hstack([size[:, :2], size[:,5:], size[:,2:5]])
                    models = np.hstack([models[:2], models[5:], models[2:5]])
                elif key=='Attention-layers':
                    mean = np.hstack([mean[:, :1], mean[:,4:], mean[:,1:4]])
                    third_quartile = np.hstack([third_quartile[:, :1], third_quartile[:,4:], third_quartile[:,1:4]])
                    maximum = np.hstack([maximum[:, :1], maximum[:,4:], maximum[:,1:4]])
                    size = np.hstack([size[:, :1], size[:,4:], size[:,1:4]])
                    models = np.hstack([models[:1], models[4:], models[1:4]])
                elif key=='Specific-attention-heads': 
                    mean = mean[:, attention_head_reordering]
                    third_quartile = third_quartile[:, attention_head_reordering]
                    maximum = maximum[:, attention_head_reordering]
                    size = size[:, attention_head_reordering]
                    models = models[attention_head_reordering]
                result[key].append({
                    'maximum': maximum,
                    'third_quartile': third_quartile,
                    'mean': mean,
                    'models': models,
                    'labels': labels,
                    'plot_name': plot_name,
                    'size': size
                })
    return result

def get_voxel_wise_max_img_on_surf(
    masker, 
    model_names, 
    language='english', 
    object_of_interest='Pearson_coeff', 
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/',
    **kwargs
    ):
    maps = []
    fsaverage = datasets.fetch_surf_fsaverage()
    for model_name in model_names:
        path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, model_name))
        name = '{}_group_fdr_effect'.format(object_of_interest)
        maps.append(fetch_map(path, name)[0])
    
    arrays = [vol_to_surf(map_, fsaverage[kwargs['surf_mesh']]) for map_ in maps]

    data_tmp = np.vstack(arrays)
    img = np.argmax(data_tmp, axis=0)
    return img

def extract_model_data(masker, data, object_of_interest, name, label, threshold_img=None, voxels_filter=None):
    result = []
    for subject in data.keys():
        try:
            array = masker.transform(data[subject][object_of_interest])
            if threshold_img is not None:
                threshold = masker.transform(threshold_img)
                if voxels_filter is not None:
                    mask = threshold >= voxels_filter
                    array = array[mask]
                    threshold = threshold[mask]
                threshold[threshold==0.0] = 10**8 # due to some voxel equal to 0 in threshold image we have nan value
                array = np.divide(array, threshold)
            mean = np.mean(array)
            median = np.median(array)
            third_quartile = np.percentile(array, 75)
            size = len(array.reshape(-1))
        except ValueError:
            mean = np.nan
            median = np.nan
            third_quartile = np.nan
            size = np.nan
        result.append([name, subject, label, mean, median, third_quartile, size])
    return result

def prepare_data_for_anova(
    model_names, 
    atlas_maps, 
    labels, 
    global_mask,
    threshold_img=None,
    voxels_filter=None,
    object_of_interest='R2', 
    language='english', 
    OUTPUT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english',
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    print("\tLooping through labeled masks...")
    result = []
    data = {}
    for model_name in model_names:
        name = model_name.replace('_{}', '')
        data[name] = {}
        for subject_id in possible_subjects_id(language):
            subject = get_subject_name(subject_id)
            path_to_map = os.path.join(OUTPUT_PATH, subject, model_name.format(subject_id))
            data[name][subject] = {
                object_of_interest: fetch_map(path_to_map, object_of_interest)[0]
            }

    for index_mask in tqdm(range(len(labels))):
        masker = get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask, PROJECT_PATH=PROJECT_PATH)
        result.append(Parallel(n_jobs=-2)(delayed(extract_model_data)(masker, data[name], object_of_interest, name, labels[index_mask], threshold_img=threshold_img, voxels_filter=voxels_filter) for name in data.keys()))
    result = [item for sublist_mask in result for subsublist_model in sublist_mask for item in subsublist_model]
    result = pd.DataFrame(result, columns=['model', 'subject', 'ROI', '{}_mean'.format(object_of_interest), '{}_median'.format(object_of_interest), '{}_3rd_quartile'.format(object_of_interest), '{}_size'.format(object_of_interest)])

    return result

def process_fmri_data(fmri_paths, masker):
    """ Load fMRI data and mask it with a given masker.
    Preprocess it to avoid NaN value when using Pearson
    Correlation coefficients in the following analysis.
    Arguments:
        - fmri_paths: list (of string)
        - masker: NiftiMasker object
    Returns:
        - data: list of length #runs (np.array of shape: #scans * #voxels) 
    """
    data = [masker.transform(f) for f in fmri_paths]
    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    for run in range(len(data)):
        zero = np.zeros(data[run].shape[0])
        new = zero.copy()
        new[0] += np.random.random()/1000
        data[run] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, data[run])
    return data
    

#########################################
############ Plot functions #############
#########################################

def plot_distribution(variable_name, language, output_path, model_names, masker, vmin=-0.4, vmax=1, xmin=-0.1, xmax=0.5, hrf='spm'):
    """ Plot the histogram of the distribution of ’variable_name’ over all subjects
    for the given language over the specific region covered by the masker.
    """
    imgs = []
    subject_list = [get_subject_name(id_) for id_ in possible_subjects_id(language)]
    for subject in subject_list:
        paths = [os.path.join(output_path, subject, model_name) for model_name in model_names]
        imgs += [fetch_map(path, variable_name)[0] for path in paths]
    data = np.hstack([masker.transform(img).reshape(-1) for img in imgs])
    print(f"""
    ##############################################################################
    ###################         HRF {hrf}        ###################
    max: {np.max(data)}
    75%: {np.percentile(data, 75)}
    median: {np.percentile(data, 50)}
    mean: {np.mean(data)}
    25%: {np.percentile(data, 25)}
    min: {np.min(data)}
    
    Data > 0.2: {100*np.sum(data>0.2)/len(data)}
    Data > 0.15: {100*np.sum(data>0.15)/len(data)}
    Data > 0.1: {100*np.sum(data>0.1)/len(data)}
    Data > 0.05: {100*np.sum(data>0.05)/len(data)}
    
    ##############################################################################
    """)
    data[data>vmax] = np.nan
    data[data<vmin] = np.nan
    display = plt.hist(data[~np.isnan(data)], bins=100)
    plt.xlim((xmin, xmax))
    plt.title(' - '.join(model_names))
    plt.savefig(display, format="png")
    plt.show()

def vertical_plot(
    data, 
    x_names, 
    analysis_name, 
    save_folder, 
    object_of_interest, 
    legend_names, 
    syntactic_roi, 
    language_roi, 
    figsize=(9,18), 
    count=False, 
    title=None, 
    ylabel='Regions of interest (ROI)', 
    xlabel='Pearson coefficients', 
    model_name='',
    percentage=False,
    x_limit=None,
    y_limit=None
    ):
    """Plots models vertically.
    """
    if percentage:
        data = data * 100
    surnames = load_surnames()
    dash_inf = 0
    x = x_names.copy()
    plt.figure(figsize=figsize) # (7.6,12)
    ax = plt.axes()
    order = np.argsort(np.mean(data, axis=1))
    data = data[order, :]
    x = [surnames[x[i]] for i in order]
    # set possible colors
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,data.shape[1] + 1)]
    for col in range(data.shape[1]):
        plt.plot(data[:,col], x, '.', alpha=0.7, markersize=9, color=colors[col])
    plt.title(title)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    if count:
        plt.axvline(x=100, color='k', linestyle='-', alpha=0.2)
    for col in range(data.shape[1]):
        plt.hlines(x, dash_inf, data[:,col], linestyle="dashed", alpha=0.05)
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.minorticks_on()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='x')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='x')
    plt.legend(legend_names, ncol=3, bbox_to_anchor=(0,0,1,1), fontsize=5)
    for index, label in enumerate(x):
        if label in language_roi:
            if label in syntactic_roi:
                ax.get_yticklabels()[index].set_bbox(dict(facecolor="green", alpha=0.4)) # set box around label
            #ax.axhspan(index-0.5, index+0.5, alpha=0.1, color='red') #set shade over the line of interest
    plt.tight_layout()
    check_folder(save_folder)
    if save_folder:
        plt.savefig(os.path.join(save_folder, '{model_name}-{analysis_name}.png'.format(model_name=model_name,
                                                                                        analysis_name=analysis_name)))
        plt.close('all')
    else:
        plt.show()

def horizontal_plot(
    data, 
    model_names, 
    analysis_name,  
    roi_names,
    title=None,
    figsize=(9,20), 
    save_folder=None,
    ylabel='', 
    xlabel='', 
    plot_name='',
    percentage=False,
    rotation=0.,
    x_limit=None,
    y_limit=None
    ):
    """Plots models horizontally.
    """
    surnames = load_surnames()
    plt.figure(figsize=figsize) # (7.6,12)
    ax = plt.axes()
    order = np.argsort(np.mean(data, axis=1))
    roi_names = [surnames[roi_names[i]] for i in order]
    if percentage:
        data = data * 100
    data = data[order, :]
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1, data.shape[0] + 1)]
    for col in range(data.shape[0]):
        plt.plot(model_names, data[col, :], '.-', alpha=0.7, markersize=9, color=colors[col])
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.minorticks_on()
    ax.tick_params(axis='x', labelsize=12, rotation=rotation)
    ax.tick_params(axis='y', labelsize=12)
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='y')
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='x')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='y')
    plt.legend(roi_names, ncol=3, bbox_to_anchor=(0,0,1,1), fontsize=10)

    plt.tight_layout()
    check_folder(save_folder)
    plt.title(title)
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, '{plot_name}-{analysis_name}.png'.format(plot_name=plot_name,
                                                                                        analysis_name=analysis_name)))
    else:
        plt.show()
    plt.close('all')

def clever_plot(
    data, 
    labels, 
    model_names, 
    save_folder=None, 
    roi_filter=load_syntactic_roi(), 
    analysis_name='',
    title='',
    ylabel='',
    xlabel='',
    plot_name='Model_comparison',
    figsize=(15,12),
    percentage=False,
    rotation=0.,
    x_limit=None,
    y_limit=None
):
    data_filtered = []
    # For syntactic ROIs
    legends = []
    for name in roi_filter:
        if name in labels:
            data_filtered.append(data[labels.index(name), :])
            legends.append(name)
    data_filtered = np.vstack(data_filtered)
    horizontal_plot(
        data_filtered, 
        model_names, 
        analysis_name=analysis_name, 
        save_folder=save_folder, 
        roi_names=legends,
        title=title,
        figsize=figsize, 
        ylabel=ylabel, 
        xlabel=xlabel, 
        plot_name=plot_name,
        percentage=percentage,
        rotation=rotation,
        x_limit=x_limit,
        y_limit=y_limit
        )

def interactive_surf_plot(surf_img, inflated=False, cmap='gist_ncar', compute_surf=True, symmetric_cmap=False, **kwargs):
    fsaverage = datasets.fetch_surf_fsaverage()
    if compute_surf:
        surf_img = vol_to_surf(surf_img, fsaverage[kwargs['surf_mesh']])
    if inflated:
        kwargs['surf_mesh'] = 'infl_left' if 'left' in kwargs['surf_mesh_type'] else 'infl_right'  
    plotting.view_surf(
        fsaverage[kwargs['surf_mesh']], 
        surf_img,
        cmap=cmap, 
        symmetric_cmap=symmetric_cmap,
        colorbar=True
        )
    plotting.show()

def cluster_map(path_to_beta_maps, title='', saving_path=None):
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps)
    else:
        data = path_to_beta_maps
    plt.figure(figsize=(40,40))
    sns.clustermap(data, row_cluster=False, method='ward', metric='cosine')
    plt.title('Heatmap -' + title)
    if saving_path is not None:
        plt.savefig(saving_path)
        plt.close()
    else:
        plt.show()


#########################################
############ Stats functions ############
#########################################

def create_one_sample_t_test(
    name, 
    maps, 
    output_dir, 
    smoothing_fwhm=None, 
    vmax=None, 
    design_matrix=None, 
    p_val=0.001, 
    fdr=0.01, 
    fwhm=6
    ):
    """ Do a one sample t-test over the maps.
    """
    print('##### ', name, ' #####')
    model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, n_jobs=-1)
    design_matrix = design_matrix if (design_matrix is not None) else pd.DataFrame([1] * len(maps),
                                                             columns=['intercept'])
    model = model.fit(maps,
                      design_matrix=design_matrix)
    z_map = model.compute_contrast(output_type='z_score')
    p_val = p_val
    z_th = norm.isf(p_val)  # 3.09
    display = plotting.plot_glass_brain(
        z_map,
        threshold=z_th,
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=name + ' z values (P<.001 unc.)',
        vmax=10)
    if output_dir:
        nib.save(z_map, os.path.join(output_dir, "{}_group_zmap.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_group_zmap".format(name)))
    else:
        plotting.show()
    # apply fdr to zmap
    thresholded_zmap, th = map_threshold(stat_img=z_map,
                                         alpha=fdr,
                                         height_control='fdr',
                                         cluster_threshold=0)
    display = plotting.plot_glass_brain(
        thresholded_zmap,
        threshold=th,
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=name + ' z values (Pfdr<.01: Z >{}'.format(np.round(th,2)),
        vmax=10)
    if output_dir:
        nib.save(thresholded_zmap, os.path.join(output_dir, "{}_group_fdr_zmap.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_fdr_group_zmap".format(name)))
    else:
        plotting.show()
    # effect size-map
    eff_map = model.compute_contrast(output_type='effect_size')

    display = plotting.plot_glass_brain(
        smooth_img(eff_map, fwhm=fwhm),
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=name + ' - Effect size ', 
        vmax=vmax)
    if output_dir:
        nib.save(eff_map, os.path.join(output_dir, "{}_group_effect.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_group_effect".format(name)))
    else:
        plotting.show()
    #thr = thresholded_zmap.get_data()
    thr = np.abs(z_map.get_data())
    eff = eff_map.get_data()
    thr_eff = eff * (thr > z_th) # 3.09
    eff_thr_map = new_img_like(eff_map, thr_eff)
    display = plotting.plot_glass_brain(
        smooth_img(eff_thr_map, fwhm=fwhm),
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=name + ' - Effect size (Pfdr<{}'.format(p_val),
        vmax=vmax)
    if output_dir:
        nib.save(eff_thr_map, os.path.join(output_dir, "{}_group_fdr_effect.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_group_fdr_effect".format(name)))
    else:
        plotting.show()
    plt.close('all')
    return new_img_like(eff_map, (thr > z_th)), (thr > z_th)

def compute_t_test_for_layer_analysis(
    data, 
    smoothing_fwhm=None, 
    analysis=None, 
    language='english',
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    if analysis is None:
        analysis = get_default_analysis()
    for model_name in data.keys():
        for analysis_of_interest in analysis.keys():
            for key in tqdm(data[model_name][analysis_of_interest].keys()):
                if key != 'models':
                    # R2
                    imgs = data[model_name][analysis_of_interest][key]['R2']
                    name = '_'.join([model_name, key])
                    output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
                    check_folder(output_dir)
                    create_one_sample_t_test(name + '_R2', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=None)
                    # Pearson coefficient
                    imgs = data[model_name][analysis_of_interest][key]['Pearson_coeff']
                    create_one_sample_t_test(name + '_Pearson_coeff', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=None)


def compute_model_contrasts_t_test(
    model1_paths,
    model2_paths,
    model1_name, 
    model2_name, 
    analysis_name,
    observed_data='Pearson_coeff',
    language='english',
    smoothing_fwhm=6,
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
     ):
    """Computer t test group analysis for ’model = model1 - model2’.
    """
    imgs = []
    for img1, img2 in zip(model1_paths, model2_paths):
        i1 = nib.load(img1)
        i2 = nib.load(img2)
        imgs.append(new_img_like(i1, i1.get_data() - i2.get_data()))
    name = '{}-vs-{}'.format(model1_name, model2_name)
    if analysis_name != '':
        name += '_' + analysis_name
    output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
    check_folder(output_dir)
    threshold_img, mask = create_one_sample_t_test(name + '_' + observed_data, imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=None)
    return threshold_img, mask


def compute_t_test_for_model_comparison(
    data, 
    name,
    smoothing_fwhm=None, 
    language='english',
    vmax=None,
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    # R2
    name = name.replace('_{}', '')
    imgs = data[name]['R2']
    output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
    check_folder(output_dir)
    name_tmp = name.replace('.', '')
    _, _ = create_one_sample_t_test(name_tmp + '_R2', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=vmax)
    # Pearson coefficient
    imgs = data[name]['Pearson_coeff']
    threshold_img, mask = create_one_sample_t_test(name_tmp + '_Pearson_coeff', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=vmax)
    return threshold_img, mask


def inter_subject_correlation(
    masker, 
    language, 
    saving_path="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/inter_subject_correlations",
    path_to_fmri_data="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI"
    ):
    subjects = [get_subject_name(subject) for subject in possible_subjects_id(language)]
    iterator = itertools.combinations(subjects, 2)
    result = {}
    for sub1, sub2 in tqdm(iterator):
        _, fMRI_paths1 = fetch_data(path_to_fmri_data, '', 
                                    sub1, language)
        _, fMRI_paths2 = fetch_data(path_to_fmri_data, '', 
                                    sub2, language)
        data1 = process_fmri_data(fMRI_paths1, masker)
        data2 = process_fmri_data(fMRI_paths2, masker)
        key = '{}_{}'.format(sub1, sub2)
        result[key] = check_correlation(data1, data2)
        np.save(os.path.join(saving_path, key + '.npy'), result[key])
    return result

def inter_subject_correlation_all_vs_1(
    masker, 
    language, 
    saving_path="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/inter_subject_correlations",
    path_to_fmri_data="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI"
    ):
    subjects = [get_subject_name(subject) for subject in possible_subjects_id(language)]
    result = {}
    data = [process_fmri_data(fetch_data(path_to_fmri_data, '', sub, language)[1], masker) for sub in subjects]
    for index, data_sub in tqdm(enumerate(data)):
        data_tmp = data[:index]+data[index+1:] if index < (len(data)-1) else data[:-1]
        data_rest = zip(*data_tmp)
        data_rest = [np.mean(np.stack(d, axis=0), axis=0) for d in list(data_rest)]
        key = '{}'.format(subjects[index])
        result[key] = check_correlation(data_sub, data_rest)
        np.save(os.path.join(saving_path, key + '.npy'), result[key])
    return result


def check_correlation(data1, data2):
    pearson_corr = []
    for index, matrix in enumerate(data1):
        pearson_corr.append(np.array([pearsonr(matrix[:,i], data2[index][:,i])[0] for i in range(matrix.shape[1])]))
    return np.stack(pearson_corr, axis=0)

def explained_variance(X, plot_name, n_components_max=1000, n_components_to_plot=[], saving_path=None):
    """Plot the explained variance of PCA dimensionality reduction.
    Arguments:
    - X: list of matrices
    - n_components_max: int
    """
    X_all = np.vstack(X)
    pca = PCA(n_components=n_components_max)
    pca.fit(X_all)
    # plot explained variance
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=n_components_max)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    plt.figure(figsize=(15,15))
    plt.plot(100 * np.cumsum(pca.explained_variance_ratio_), color='black')
    plt.xlabel('eigenvalue number')
    plt.ylabel('explained variance (%)')
    variances = []
    for n_components in n_components_to_plot:
        var_model = 100 * sum(pca.explained_variance_ratio_[:n_components])
        variances.append(var_model)
        plt.axhline(y=var_model, color=scalarMap.to_rgba(n_components), linestyle='--', label='{}% variance explained with {} components'.format(round(var_model, 1), n_components))
        plt.axvline(x=n_components, color=scalarMap.to_rgba(n_components), linestyle='--')
    plt.xticks(list(plt.xticks()[0]) + n_components_to_plot)
    plt.yticks(list(plt.xticks()[0]) + variances)
    plt.title('\n'.join(wrap(plot_name)))
    plt.legend()
    plt.xlim((0, n_components_max))
    plt.ylim((0, 100))
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    else:
        plt.show()
    plt.close()



#########################################
############ Voxel Clustering ###########
#########################################

def heat_map(path_to_beta_maps, aggregate=False, title='', saving_path=None):
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps)
    else:
        data = path_to_beta_maps
    plt.figure(figsize=(40,40))
    if aggregate:
        data = aggregate_beta_maps(data, nb_layers=13, layer_size=768)
    sns.heatmap(data)
    plt.title('Heatmap -' + title)
    if saving_path is not None:
        plt.savefig(saving_path)
        plt.close()
    else:
        plt.show()

def resample_beta_maps(path_to_beta_maps, original_masker, new_masker):
    """Resample beta map with a new masker.
    path_to_beta_maps: #voxels x #features"""
    if isinstance(original_masker, str):
        original_masker = load_masker(original_masker)
    if isinstance(new_masker, str):
        new_masker = load_masker(new_masker)
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps)
    else:
        data = path_to_beta_maps
    print("Original data has dimension: ", data.shape)
    imgs = original_masker.inverse_transform([data[:, i] for i in range(data.shape[1])])
    new_data = new_masker.transform(imgs) # dimension: #samples x #voxels
    print("New data has dimension: ", new_data.T.shape)
    return new_data.T

def create_subject_mask(mask_template, subject_name, subject_id, model_name, global_masker, threshold=75):
    mask_img = nib.load(mask_template.format(model_name=model_name, subject_name=subject_name, subject_id=subject_id))
    mask_tmp = global_masker.transform(mask_img)
    mask = np.zeros(mask_tmp.shape)
    mask[mask_tmp > np.percentile(mask_tmp, threshold)] = 1
    mask = mask.astype(int)
    mask = mask[0].astype(bool)
    return mask

def aggregate_beta_maps(data, nb_layers=13, layer_size=768, attention_head_size=64, nb_attention_heads=12, aggregation_type='layer', n_components=100):
    """Clustering of voxels or ROI based on their beta maps.
    maps: (#voxels x #features)
    """
    if isinstance(data, str):
        data = np.load(data)
    nb_layers = nb_layers
    layer_size = layer_size
    if aggregation_type=='layer':
        result = np.zeros((data.shape[0], nb_layers))
        for index in range(nb_layers):
            result[:, index] = np.mean(data[:, layer_size * index : layer_size * (index + 1)], axis=1)
    elif aggregation_type=='pca':
        result = PCA(n_components=n_components, random_state=1111).fit_transform(data)
    elif aggregation_type=='umap':
        result = np.hstack([umap.UMAP(n_neighbors=5, min_dist=0., n_components=n_components, random_state=1111, metric='cosine', output_metric='euclidean').fit_transform(data[:, layer_size * index : layer_size * (index + 1)]) for index in range(nb_layers)])
    elif aggregation_type=='attention_head':
        result = np.zeros((data.shape[0], (nb_layers-1)*nb_attention_heads))
        for index in range((nb_layers-1)*nb_attention_heads):
            result[:, index] = np.mean(data[:, 768 + attention_head_size * index : 768 + attention_head_size * (index + 1)], axis=1)
    return result

def plot_roi_img_surf(surf_img, saving_path, plot_name, mask=None, labels=None, inflated=False, compute_surf=True, colorbar=True, return_plot=False, **kwargs):
    fsaverage = datasets.fetch_surf_fsaverage()
    if compute_surf:
        surf_img = vol_to_surf(surf_img, fsaverage[kwargs['surf_mesh']], interpolation='nearest', mask_img=mask)
        surf_img[np.isnan(surf_img)] = -1
    if labels is not None:
        surf_img = np.round(surf_img)
        
    surf_img += 1
    if saving_path:
        plt.close('all')
        plt.hist(surf_img, bins=50)
        plt.title(plot_name)
        plt.savefig(saving_path + "hist_{}.png".format(plot_name))
    #plt.close('all')
    if inflated:
        kwargs['surf_mesh'] = 'infl_left' if 'left' in kwargs['surf_mesh_type'] else 'infl_right' 
    disp = plotting.plot_surf_roi(
        surf_mesh=fsaverage[kwargs['surf_mesh']], 
        roi_map=surf_img,
        hemi=kwargs['hemi'],
        view=kwargs['view'],
        bg_map=fsaverage[kwargs['bg_map']], 
        bg_on_data=kwargs['bg_on_data'],
        darkness=kwargs['darkness'],
        colorbar=colorbar,
        axes=kwargs['axes'],
        figure=kwargs['figure'])
    if saving_path:
        disp.savefig(saving_path + plot_name + '_{}_{}_{}.png'.format(kwargs['surf_mesh_type'], kwargs['hemi'], kwargs['view']))
    if return_plot:
        plotting.show()
        return disp
    plt.close('all')

def scatter3d(data, cs, colorsMap='jet', reduction_type='', other=None, **kwargs):
    """3D scatter plot with random color to distinguish the shape."""
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', c=scalarMap.to_rgba(cs), **kwargs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(other['elevation'], other['azimuth'])

    #ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.title('Activations embedding for {}'.format(reduction_type), fontsize=24)
    ax.view_init(other['elevation'], other['azimuth'])
    interact()

def plot_reduction(data, plot_type='2D', reduction_type='', other=None, **kwargs):
    """kwargs includes: 
     - s=5
     - c='density'
     - cmap='Spectral'
    """
    plt.close('all')
    if plot_type=='2D':
        plt.scatter(data[:, 0], data[:, 1], **kwargs)
        interact()
    elif plot_type=='3D':
        scatter3d(data, data[:, 2], colorsMap='jet', reduction_type=reduction_type, other=other, **kwargs)

def plot_reduc(path_to_beta_maps, n_neighbors=30, min_dist=0.0, n_components=3, random_state=1111, cluster_selection_epsilon=0.0):
    """Clustering of voxels or ROI based on their beta maps.
    """
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps).T
    else:
        data = path_to_beta_maps.T
    
    plot_kwargs = {
        's':2,
        #'c':'density',
        'cmap':'Spectral'
    }
    data = data.T
    print("UMAP...")
    umap_result_3D = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, metric='manhattan').fit_transform(data)
    plot_reduction(umap_result_3D, plot_type='3D', reduction_type='UMAP', other={'elevation': 15, 'azimuth': 15}, **plot_kwargs)
    
def reduce(data, reduction, n_neighbors=30, min_dist=0.0, n_components=10, random_state=1111, affinity_reduc='nearest_neighbors', metric_r='cosine', output_metric='euclidean', mapper=None, mapper_plot=None):
    """Reduce dimemnsion of beta maps.
    """
    gc.collect()
    if mapper is not None:
        data_reduced = mapper.transform(data)
        standard_embedding = mapper_plot.transform(data)
    elif reduction=="umap":
        mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, metric=metric_r, output_metric=output_metric).fit(data)
        mapper_plot = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state, metric=metric_r, output_metric=output_metric).fit(data)
        data_reduced = mapper.transform(data)
        standard_embedding = mapper_plot.transform(data)
    elif reduction=="pca":
        mapper = PCA(n_components=n_components, random_state=random_state).fit(data)
        mapper_plot = PCA(n_components=2, random_state=random_state).fit(data)
        data_reduced = mapper.transform(data)
        standard_embedding = mapper_plot.transform(data)
    elif reduction=="ica":
        mapper = FastICA(n_components=n_components, random_state=random_state).fit(data)
        mapper_plot = FastICA(n_components=2, random_state=random_state).fit(data)
        data_reduced = mapper.transform(data)
        standard_embedding = mapper_plot.transform(data)
    else:
        data_reduced = data
        mapper = None
        mapper_plot = None
        standard_embedding = None
    return data_reduced, standard_embedding, mapper, mapper_plot

def cluster(data_reduced, mask, clustering, min_cluster_size=20, min_samples=10, n_clusters=13, random_state=1111, cluster_selection_epsilon=0.0, affinity_cluster='cosine', linkage='average', saving_folder=None, plot_name=None, metric_c='euclidean', memory=None, predictor=None):
    """Clustering of voxels or ROI based on their beta maps.
    """ 
    gc.collect()
    if predictor is not None:
        labels_, proba = hdbscan.approximate_predict(predictor, data_reduced)
    elif clustering=='hdbscan':
        predictor = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, metric=metric_c, prediction_data=True).fit(data_reduced)
        labels_, proba = hdbscan.approximate_predict(predictor, data_reduced)
    #elif clustering=='agg-clus':
    #    #kneighbors_graph_ = kneighbors_graph(data_, n_neighbors=5)
    #    sc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity_cluster, memory=memory) #  connectivity=kneighbors_graph_
    #    clustering = sc.fit(data_reduced)
    #    labels_ = clustering.labels_
    #elif clustering=="max":
    #    labels_ = np.argmax(data_reduced, axis=1)
    if mask is not None:
        all_labels = np.zeros(len(mask))
        all_labels[mask] = labels_
        all_labels[~mask] = -1
    else:
        all_labels = labels_
    plot_name += '_nb_clusters-{}'.format(len(np.unique(all_labels)))
    return all_labels, labels_, plot_name, predictor

def plot_scatter(labels, standard_embedding, plot_name, saving_folder, return_plot=False):
    # Printing scatter plot
    plt.close('all')
    fig = plt.figure(figsize=(15, 15))
    clustered = (labels >= 0)
    colorsMap='jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(labels), vmax=max(labels))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.scatter(standard_embedding[clustered, 0],
                standard_embedding[clustered, 1],
                c=scalarMap.to_rgba(labels[clustered]),
                s=0.1,
                cmap='Spectral')
    plt.title(plot_name)
    check_folder(saving_folder)
    if saving_folder is not None:
        plt.savefig(saving_folder + "scatter_{}.png".format(plot_name), dpi=300)
    if return_plot:
        plt.show()
    plt.close('all')

def plot_text(labels, standard_embedding, onsets, plot_name, saving_folder, return_plot=False):
    # Printing scatter plot
    plt.close('all')
    fig = plt.figure(figsize=(30, 30))
    clustered = (labels >= 0)
    colorsMap='jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(labels), vmax=max(labels))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    for a,b,c,d in zip(standard_embedding[clustered, 0], standard_embedding[clustered, 1], onsets, scalarMap.to_rgba(labels[clustered])):

        plt.text(
            a,
            b,
            c,
            c=d, 
            size=1,
            wrap=True#s=0.1,
        )
    plt.xlim((-5, 25))
    plt.ylim((-5, 25))
    plt.tight_layout()
    plt.title(plot_name)
    check_folder(saving_folder)
    if saving_folder is not None:
        plt.savefig(saving_folder + "text_{}.png".format(plot_name), dpi=1000)
    if return_plot:
        plt.show()
    plt.close('all')

def create_name(subject_name, params):
    keys = [
            'reduction',
            'clustering',
            'min_cluster_size',
            'n_neighbors',
            'n_components',
            'affinity_cluster',
            'linkage',
            'min_samples',
            'metric_r',
            'metric_c',
            'output_metric'
    ]
    keys_surname = [
            'red',
            'clust',
            'min-c-size',
            'n-neigh',
            'n-comp',
            'aff-clus',
            'link',
            'min-samples',
            'metric-r',
            'metric-c',
            'out-metric'
    ]
    name = subject_name
    for index, key in enumerate(keys):
        name += '_' + keys_surname[index] + '-' + str(params[key])
    return name

def save_results(data, all_original_length, saving_folder, plot_name, global_masker, mask, return_plot=False, subject_wise=True, inflated=False, **kwargs):
    """Save surface plot."""
    gc.collect()
    if isinstance(data, str):
        data = np.load(data)
    elif saving_folder is not None:
        check_folder(os.path.join(saving_folder, 'labels'))
        np.save(os.path.join(saving_folder, 'labels', plot_name+'.npy'), data)
    if subject_wise: # data computed on single subject
        img = global_masker.inverse_transform(data)
        plot_roi_img_surf(img, saving_folder, plot_name, mask=mask, labels='round', inflated=inflated, compute_surf=True, return_plot=return_plot, colorbar=True, **kwargs)
    else: # data computed on all subjects
        i = 0
        results = []
        nb_clusters = len(np.unique(data))
        print('Input shape: ', data.shape)
        print('Number of clusters: ', nb_clusters)
        for l in all_original_length:
            results.append(get_prob_matrix(data[i: i+l], nb_clusters))
            i += l
        l = results[0].shape[1]
        print('Number of voxels:', l)
        for index, item in enumerate(results):
            assert item.shape[1]==l
        matrix_ = np.stack(results, axis=0)
        prob_matrix = np.mean(matrix_, axis=0)
        argmax_matrix = np.argmax(prob_matrix, axis=0)
        #final_matrix = np.zeros((nb_clusters, len(argmax_matrix)))
        #for index, i in enumerate(rois):
        #    final_matrix[i, index] = 1
        print('Effective number of clusters:', len(np.unique(argmax_matrix)))
        
        img = global_masker.inverse_transform(argmax_matrix)
        # Cleaning memory
        gc.collect()
        del matrix_
        del prob_matrix
        del argmax_matrix
        del results
        plot_roi_img_surf(img, saving_folder, plot_name, mask=mask, labels='round', inflated=False, compute_surf=True, return_plot=return_plot, colorbar=True, **kwargs)
        #plot_roi_img_surf(img, saving_folder, plot_name, mask=None, labels=np.unique(results), inflated=False, compute_surf=True, colorbar=True, **kwargs)

def get_connectivity_matrix(labels):
    if isinstance(labels, str):
        labels = np.load(labels)
    result = np.equal.outer(labels,labels).astype(int)
    return result

def get_prob_matrix(labels, nb_clusters):
    if isinstance(labels, str):
        labels = np.load(labels)
    nb_voxels = len(labels)
    prob_matrix = np.zeros((nb_clusters, nb_voxels))
    for index, i in enumerate(range(nb_clusters)):
        prob_matrix[index, labels==i] = 1
    return prob_matrix

def get_labels(subject_name, reduction, clustering, saving_folder, params):
    """Retrieve computed labels for a given subject."""
    saving_folder = os.path.join(saving_folder, '_'.join([reduction, clustering]) + '/')
    plot_name = create_name(subject_name, params)
    path = sorted(glob.glob(os.path.join(saving_folder, 'labels', plot_name + '*.npy')))
    return path

def get_images(subject_name, reduction, clustering, saving_folder, params):
    """Retrieve computed labels for a given subject."""
    saving_folder = os.path.join(saving_folder, '_'.join([reduction, clustering]) + '/')
    plot_name = create_name(subject_name, params)
    path = sorted(glob.glob(os.path.join(saving_folder, plot_name + '*.png')))
    return path

def extract_indexes(list_, item):
    return np.where(list_==item)

def extract_clusters(labels_, onsets):
    result = {}
    for value in np.unique(labels_):
        result[value] = onsets[np.where(labels_==value)]
    return result

def plot_functional_clustering(
    data, 
    mask,
    masker,
    average_mask,
    nb_voxels,
    nb_subjects,
    data_name,
    params,
    saving_folder,
    n_neighbors_list=[3, 4, 5],
    min_samples_list=[5, 8, 10, 12, 15, 17, 20, 23,  25, 27, 30],
    min_cluster_size_list=[200, 300, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1350 ,1400, 1450, 1500, 1550, 1600, 1700, 1750, 1800, 1850, 1900],
    tmp_folder='', 
    reduction='umap',
    clustering='hdbscan',
    metric_reduction='cosine',
    metric_clustering='euclidean',
    output_metric='euclidean',
    random_state=1111,
    min_dist=0.,
    affinity_reduc='nearest_neighbors',
    n_components=3,
    cluster_selection_epsilon=0.,
    subject_id=None,
    return_plot=False,
    **kwargs
    ):
    """Plot functional clustering of voxels determined over a set of subjects.
    Arguments:
    - data: np.array (2D - size #subects*#voxels x #features)
    - mask:np.array (1D - size #subects*#voxels)
    - masker: nifti-masker
    - average_mask: nifti image
    - nb_voxels: int
    - nb_subjects: int
    - data_name: str
    - params: dict
    - n_neighbors_list: list of int
    - reduction: str
    - clustering: str
    - metric_reduction: str
    - metric_clustering: str
    - random_state: int
    - min_dist: float
    - affinity_reduc: str
    - n_components: int
    - cluster_selection_epsilon: float

    Returns:
    - display plot(s)
    """

    for n_neighbors in n_neighbors_list:
        
        if os.path.exists(os.path.join(tmp_folder, f'data_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy')):
            print('Loading...')
            data_reduced = np.load(os.path.join(tmp_folder, f'data_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'))
            standard_embedding = np.load(os.path.join(tmp_folder, f'standard_embedding_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'))
            mapper = load(os.path.join(tmp_folder, f'mapper_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
            mapper_plot = load(os.path.join(tmp_folder, f'mapper_plot_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
        else:
            print('Reducing...')
            data_reduced, standard_embedding, mapper, mapper_plot = reduce(
                data, 
                reduction, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components, 
                random_state=random_state, 
                affinity_reduc=affinity_reduc, 
                metric_r=metric_reduction, 
                output_metric=output_metric)
            np.save(os.path.join(tmp_folder, f'data_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'), data_reduced)
            np.save(os.path.join(tmp_folder, f'standard_embedding_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'), standard_embedding)
            dump(mapper, os.path.join(tmp_folder, f'mapper_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
            dump(mapper_plot, os.path.join(tmp_folder, f'mapper_plot_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
                    
        saving_folder_ = os.path.join(saving_folder, 'data-{}_'.format(data_name) + '_'.join([reduction, clustering]) + '/')
                
        for min_samples in min_samples_list:

            for min_cluster_size in min_cluster_size_list:
                params.update({
                    'reduction':reduction,
                    'clustering':clustering, 
                    'n_components':n_components,
                    'n_clusters':0,
                    'min_samples': min_samples,
                    'min_cluster_size': min_cluster_size,
                    'cluster_selection_epsilon': cluster_selection_epsilon,
                    'min_dist': min_dist,
                    'n_neighbors': n_neighbors,
                    'metric_r': metric_reduction,
                    'metric_c': metric_clustering,
                    'output_metric': output_metric
                })
                plot_name = create_name('all-subjects', params) if nb_subjects > 1 else create_name(str(subject_id), params)

                all_labels, labels_, plot_name, predictor = cluster(
                        data_reduced, 
                        mask, 
                        clustering, 
                        min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        n_clusters=0, 
                        random_state=random_state, 
                        cluster_selection_epsilon=cluster_selection_epsilon, 
                        affinity_cluster='', 
                        linkage='', 
                        saving_folder=saving_folder_, 
                        plot_name=plot_name, 
                        metric_c=metric_clustering)

                plot_scatter(labels_, standard_embedding, plot_name, saving_folder_, return_plot=return_plot)
                save_results(all_labels, (np.ones(nb_subjects)*nb_voxels).astype(int), saving_folder_, plot_name, masker, average_mask, return_plot=return_plot, subject_wise=False, **kwargs)

def plot_semantic_clustering(
    data_words,
    path_to_data_template,
    average_mask,
    original_masker,
    new_masker,
    onsets,
    subject_names_list,
    subject_ids_list,
    data_name,
    params,
    saving_folder,
    n_neighbors_list=[3, 4, 5],
    min_samples_list=[5, 8, 10, 12, 15, 17, 20, 23,  25, 27, 30],
    min_cluster_size_list=[200, 300, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200],
    tmp_folder='', 
    model_name='',
    aggregation_type='attention_head',
    reduction='umap',
    clustering='hdbscan',
    metric_reduction='cosine',
    metric_clustering='euclidean',
    output_metric='euclidean',
    random_state=1111,
    min_dist=0.,
    affinity_reduc='nearest_neighbors',
    n_components=3,
    cluster_selection_epsilon=0.,
    subject_id=None,
    threshold=90,
    nb_layers=13,
    layer_size=768,
    n_components_aggregation=100,
    return_plot=False,
    standardize_beta_maps=True,
    **kwargs
    ):
    """Plot functional clustering of voxels determined over a set of subjects.
    Arguments:
    - data_words: np.array (2D - size #words x #features)
    - path_to_data_template: str
    - average_mask: niftii image
    - original_masker: nifti-masker
    - new_masker: nifti-masker
    - onsets: np.array (1D) - list of words
    - subject_names_list: list of string
    - subject_ids_list: list of string
    - data_name: str
    - params: dict
    - saving_folder: str (folder to save results)
    - min_samples_list: list of int
    - min_cluster_size_list: list of int
    - tmp_folder: str (folder to save past computations)
    - model_name: str
    - aggregation_type: str
    - n_neighbors_list: list of int
    - reduction: str
    - clustering: str
    - metric_reduction: str
    - metric_clustering: str
    - random_state: int
    - min_dist: float
    - affinity_reduc: str
    - n_components: int
    - cluster_selection_epsilon: float
    - nb_layers: int
    - layer_size: int
    - n_components_aggregation: int (pre-reduction)
    - return_plot: bool
    - standardize_beta_maps: bool

    Returns:
    - display plot(s)
    """
    voxels_mask = new_masker.transform(average_mask)
    voxels_mask = voxels_mask.astype(int)
    voxels_mask = voxels_mask[0].astype(bool)
    
    fmri_data_dict = {}
    data_voxels_reduced_dict = {}
    standard_voxels_embedding_dict = {}
    
    def get_subject_data(path_to_data_template, subject_name, subject_id, voxels_mask,
                     original_masker, new_masker, nb_layers, layer_size, 
                     aggregation_type, n_components, standardize_beta_maps=True):
        path_to_beta_map = path_to_data_template.format(subject_name=subject_name, subject_id=subject_id)
        data_fmri = resample_beta_maps(
                            aggregate_beta_maps(
                                path_to_beta_map,
                                nb_layers=nb_layers,
                                layer_size=layer_size, 
                                aggregation_type=aggregation_type,
                                n_components=n_components_aggregation
                            ), 
                            original_masker, 
                            new_masker)[voxels_mask, :]
        # standardize beta maps if necessary
        if standardize_beta_maps:
            scaler = StandardScaler(with_mean=True, with_std=True)
            data_fmri = scaler.fit_transform(data_fmri.T).T
            
        return data_fmri
    
    data_fmri_list = Parallel(n_jobs=-2)(delayed(get_subject_data)(
            path_to_data_template, 
            subject_name, 
            subject_id,
            voxels_mask, 
            original_masker, 
            new_masker, 
            nb_layers=nb_layers, 
            layer_size=layer_size,
            aggregation_type=aggregation_type, 
            n_components=n_components_aggregation,
            standardize_beta_maps=standardize_beta_maps
        ) for subject_name, subject_id in zip(subject_names_list, subject_ids_list))
        
        
    for index_sub, subject_name in enumerate(subject_names_list):
        fmri_data_dict[subject_name] = data_fmri_list[index_sub]

    for n_neighbors in n_neighbors_list:
        
        print('Number of neighbors: ', n_neighbors)
            
        if os.path.exists(os.path.join(tmp_folder, f'data_words_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy')):
            print('Loading...')
            data_words_reduced = np.load(os.path.join(tmp_folder, f'data_words_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'))
            standard_words_embedding = np.load(os.path.join(tmp_folder, f'standard_words_embedding_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'))
            mapper = load(os.path.join(tmp_folder, f'mapper_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
            mapper_plot = load(os.path.join(tmp_folder, f'mapper_plot_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
        else:
            print('Reducing...')
            data_words_reduced, standard_words_embedding, mapper, mapper_plot = reduce(
                data_words, 
                reduction, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components, 
                random_state=random_state, 
                affinity_reduc=affinity_reduc, 
                metric_r=metric_reduction, 
                output_metric=output_metric)
            np.save(os.path.join(tmp_folder, f'data_words_reduced_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'), data_words_reduced)
            np.save(os.path.join(tmp_folder, f'standard_words_embedding_{reduction}_{n_neighbors}_{n_components}_{data_name}.npy'), standard_words_embedding)
            dump(mapper, os.path.join(tmp_folder, f'mapper_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
            dump(mapper_plot, os.path.join(tmp_folder, f'mapper_plot_{reduction}_{n_neighbors}_{n_components}_{data_name}.joblib'))
            
        for subject_name in tqdm(fmri_data_dict.keys()):
            data_fmri = fmri_data_dict[subject_name]

            data_voxels_reduced, standard_voxels_embedding, _, _ = reduce(
                data_fmri, 
                reduction, 
                mapper=mapper,
                mapper_plot=mapper_plot)
            data_voxels_reduced_dict[subject_name] = data_voxels_reduced
            standard_voxels_embedding_dict[subject_name] = standard_voxels_embedding

        saving_folder_ = os.path.join(saving_folder, data_name + '_'.join([reduction, clustering]) + '/')

        for min_samples in min_samples_list:

            for min_cluster_size in min_cluster_size_list:
                
                print('min_cluster_size: ', min_cluster_size, 'min_samples: ', min_samples)
                
                params.update({
                    'reduction':reduction,
                    'clustering':clustering, 
                    'n_components':n_components,
                    'n_clusters':0,
                    'min_samples': min_samples,
                    'min_cluster_size': min_cluster_size,
                    'cluster_selection_epsilon': cluster_selection_epsilon,
                    'min_dist': min_dist,
                    'n_neighbors': n_neighbors,
                    'metric_r': metric_reduction,
                    'metric_c': metric_clustering,
                    'output_metric': output_metric
                })
                plot_name = create_name('', params)

                _, words_labels_, plot_name, predictor = cluster(
                        data_words_reduced,
                        None, 
                        clustering, 
                        min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        n_clusters=0, 
                        random_state=random_state, 
                        cluster_selection_epsilon=cluster_selection_epsilon, 
                        affinity_cluster='', 
                        linkage='', 
                        saving_folder=saving_folder_, 
                        plot_name=plot_name, 
                        metric_c=metric_clustering)
                
                words_labels_ += 1 # because we want the '-1' labelled voxels (those that do not belong to any cluster), to appear transparent
                plot_scatter(words_labels_, standard_words_embedding, 'words_' + plot_name, saving_folder_, return_plot=return_plot)
                plot_text(words_labels_, standard_words_embedding, onsets, plot_name, saving_folder_)
                
                for subject_name in tqdm(data_voxels_reduced_dict.keys()):
                    
                    plot_name = create_name(subject_name, params)
                    
                    all_voxels_labels, voxels_labels_, plot_name, _ = cluster(
                            data_voxels_reduced_dict[subject_name],
                            voxels_mask, 
                            clustering, 
                            min_cluster_size=min_cluster_size, 
                            min_samples=min_samples, 
                            n_clusters=0, 
                            random_state=random_state, 
                            cluster_selection_epsilon=cluster_selection_epsilon, 
                            affinity_cluster='', 
                            linkage='', 
                            saving_folder=saving_folder_, 
                            plot_name=plot_name, 
                            metric_c=metric_clustering,
                            predictor=predictor)
                    
                    voxels_labels_ += 1 # idem
                    
                    plot_scatter(voxels_labels_, standard_voxels_embedding, 'voxels_' + plot_name, saving_folder_, return_plot=return_plot)
                    
                    save_results(all_voxels_labels, None, saving_folder_, plot_name, new_masker, new_masker.inverse_transform(voxels_mask), return_plot=return_plot, inflated=True,  **kwargs) 

def multi_plot(
    paths, 
    saving_path,
    nb_subjects,
    nb_voxels,
    plot_name, 
    new_masker, 
    average_mask,
    return_plot=True, 
    **kwargs
    ):
    """Plots all 4 surface views.
    """
    for path in paths:
        
        item = os.path.basename(os.path.dirname(os.path.dirname(path)))
        data_name = '_'.join(item.split('_')[:-2])
        reduction = item.split('_')[-2]
        clustering = item.split('_')[-1]
        name = '_'.join(os.path.basename(path).split('clust-')[1].split('_')[1:]).split('.')[0]
        model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        if model_name not in ['bert-base-cased_pre-7_1_post-0_norm-None_norm-inf', 'gpt2_pre-20_1_norm-inf_norm-inf']:
            model_name = 'bert-base-cased_pre-7_1_post-0_norm-None_norm-inf'
            
            
        save_in = os.path.join(saving_path, 'results', data_name, model_name, reduction, clustering)
        check_folder(save_in)
        
        plt.close('all')
        fig1 = plt.figure(figsize=(15,10))
        ax1 = fig1.add_subplot(2, 2, 1, projection= '3d')
        ax2 = fig1.add_subplot(2, 2, 2, projection= '3d')
        ax3 = fig1.add_subplot(2, 2, 3, projection= '3d')
        ax4 = fig1.add_subplot(2, 2, 4, projection= '3d')

        view = 'left' #left
        kwargs = {
            'surf_mesh': f'pial_{view}', # pial_right, infl_left, infl_right
            'surf_mesh_type': f'pial_{view}',
            'hemi': view, # right
            'view':'lateral', # medial
            'bg_map': f'sulc_{view}', # sulc_right
            'bg_on_data':True,
            'darkness':.8,
            'axes': ax1,
            'figure': fig1
        }
        save_results(path, (np.ones(nb_subjects)*nb_voxels).astype(int), None, plot_name, new_masker, average_mask, return_plot=return_plot, **kwargs) #params['saving_folder']

        view = 'left' #left
        kwargs = {
            'surf_mesh': f'pial_{view}', # pial_right, infl_left, infl_right
            'surf_mesh_type': f'pial_{view}',
            'hemi': view, # right
            'view':'medial', # medial
            'bg_map': f'sulc_{view}', # sulc_right
            'bg_on_data':True,
            'darkness':.8,
            'axes': ax2,
            'figure': fig1
        }
        save_results(path, (np.ones(nb_subjects)*nb_voxels).astype(int), None, plot_name, new_masker, average_mask, return_plot=return_plot, **kwargs) #params['saving_folder']
        
        view = 'right' #left
        kwargs = {
            'surf_mesh': f'pial_{view}', # pial_right, infl_left, infl_right
            'surf_mesh_type': f'pial_{view}',
            'hemi': view, # right
            'view':'lateral', # medial
            'bg_map': f'sulc_{view}', # sulc_right
            'bg_on_data':True,
            'darkness':.8,
            'axes': ax3,
            'figure': fig1
        }
        save_results(path, (np.ones(nb_subjects)*nb_voxels).astype(int), None, plot_name, new_masker, average_mask, return_plot=return_plot, **kwargs) #params['saving_folder']
        
        view = 'right' #left
        kwargs = {
            'surf_mesh': f'pial_{view}', # pial_right, infl_left, infl_right
            'surf_mesh_type': f'pial_{view}',
            'hemi': view, # right
            'view':'medial', # medial
            'bg_map': f'sulc_{view}', # sulc_right
            'bg_on_data':True,
            'darkness':.8,
            'axes': ax4,
            'figure': fig1
        }
        save_results(path, (np.ones(nb_subjects)*nb_voxels).astype(int), None, plot_name, new_masker, average_mask, return_plot=return_plot, **kwargs) #params['saving_folder']
        
        plt.suptitle(name)
        plt.savefig(os.path.join(save_in, name+'.png'))
        plt.close('all')