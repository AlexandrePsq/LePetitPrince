import os
import umap
import glob
import itertools
import matplotlib
import numpy as np
import pandas as pd
import pingouin as pg
from tqdm import tqdm
import matplotlib.pyplot as plt


import scipy
import hdbscan
import nistats
import nilearn
import nibabel as nib
from nilearn import plotting
from nilearn import datasets
from scipy.stats import norm
from scipy.stats import pearsonr
from nilearn.regions import RegionExtractor
from nistats.thresholding import map_threshold
from sklearn.cluster import AgglomerativeClustering
from nistats.second_level_model import SecondLevelModel
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_surf_roi
from nilearn.surface import vol_to_surf

from logger import Logger
from utils import read_yaml, check_folder, fetch_masker, possible_subjects_id, get_subject_name, fetch_data, get_roi_mask, filter_args, load_masker


SEED = 1111


#########################################
############ Basic functions ############
#########################################

def fetch_map(path, distribution_name):
    path = os.path.join(path, '*' + distribution_name+'.nii.gz')
    files = sorted(glob.glob(path))
    return files

def load_atlas(name='cort-prob-2mm'):
    atlas = datasets.fetch_atlas_harvard_oxford(name)
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
    for i in range(m//size):
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
                Pearson_coeff_maps[subject] = fetch_map(path_to_map, 'Pearson_coeff')
                print(subject, '-', len(R2_maps[subject]), '-', len(Pearson_coeff_maps[subject]))
            if analysis_of_interest == 'Hidden-layers':
                data[model_name][analysis_of_interest]['models'] = [os.path.dirname(name).split('_')[-1] for name in list(Pearson_coeff_maps.values())[-1]]
            else:
                data[model_name][analysis_of_interest]['models'] = ['-'.join(os.path.dirname(name).split('_')[-2:]) for name in list(Pearson_coeff_maps.values())[-1]]
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

def fit_per_roi(maps, atlas_maps, labels, global_mask):
    print("\tLooping through labeled masks...")
    mean = np.zeros((len(labels)-1, len(maps)))
    third_quartile = np.zeros((len(labels)-1, len(maps)))
    maximum = np.zeros((len(labels)-1, len(maps)))
    for index_mask in tqdm(range(len(labels)-1)):
        masker = get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask)
        for index_model, map_ in enumerate(maps):
            try:
                array = masker.transform(map_)
                maximum[index_mask, index_model] = np.max(array)
                third_quartile[index_mask, index_model] = np.percentile(array, 75)
                mean[index_mask, index_model] = np.mean(array)
            except ValueError:
                maximum[index_mask, index_model] = np.nan
                third_quartile[index_mask, index_model] = np.nan
                mean[index_mask, index_model] = np.nan
            
    print("\t\t-->Done")
    return mean, third_quartile, maximum

def get_data_per_roi(
    data, 
    atlas_maps,
    labels,
    global_mask=None,
    analysis=None, 
    model_name=None,
    language='english', 
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    x_labels = labels[1:]
    models = None
    if analysis is None:
        maps = []
        for model_name in data.keys():
            path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, model_name))
            name = 'R2_group_fdr_effect'
            maps.append(fetch_map(path, name)[0])
        plot_name = ["Model comparison"]
        # extract data
        mean, third_quartile, maximum = fit_per_roi(maps, atlas_maps, labels, global_mask=global_mask)
    else:
        for key in analysis:
            maps = []
            for model in data[model_name][key].keys():
                if model == 'models':
                    models = data[model_name][key][model]
                else:
                    name = '_'.join([model_name, model])
                    path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
                    name = 'R2_group_fdr_effect'
                    maps.append(fetch_map(path, name)[0])
            plot_name = ["{} for {}".format(model_name, key)]

            mean = np.zeros((len(labels)-1, len(models)))
            third_quartile = np.zeros((len(labels)-1, len(models)))
            # extract data
            mean, third_quartile, maximum = fit_per_roi(maps, atlas_maps, labels, global_mask=global_mask)
            print("\t\t-->Done")
            # Small reordering of models so that layers are in increasing order
            if key=='Hidden-layers':
                mean = np.hstack([mean[:, :2], mean[:,5:], mean[:,2:5]])
                third_quartile = np.hstack([third_quartile[:, :2], third_quartile[:,5:], third_quartile[:,2:5]])
                maximum = np.hstack([maximum[:, :2], maximum[:,5:], maximum[:,2:5]])
                models = models[:2] + models[5:] + models[2:5]
            elif key=='Attention-layers':
                mean = np.hstack([mean[:, :1], mean[:,4:], mean[:,1:4]])
                third_quartile = np.hstack([third_quartile[:, :1], third_quartile[:,4:], third_quartile[:,1:4]])
                maximum = np.hstack([maximum[:, :1], maximum[:,4:], maximum[:,1:4]])
                models = models[:1] + models[4:] + models[1:4]
            elif key=='Specific-attention-heads':
                mean = np.hstack([mean[:, :4], mean[:,6:7], mean[:,4:6], mean[:,7:]])
                third_quartile = np.hstack([third_quartile[:, :4], third_quartile[:,6:7], third_quartile[:,4:6], third_quartile[:,7:]])
                maximum = np.hstack([maximum[:, :4], maximum[:,6:7], maximum[:,4:6], maximum[:,7:]])
                models = models[:4] + models[6:7] + models[4:6] + models[7:]
    result = {
        'maximum': maximum,
        'third_quartile': third_quartile,
        'mean': mean,
        'models': models,
        'x_labels': x_labels,
        'plot_name': plot_name,
    }
    return result

def get_voxel_wise_max_img_on_surf(
    masker, 
    model_names, 
    language='english', 
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/',
    **kwargs
    ):
    maps = []
    fsaverage = datasets.fetch_surf_fsaverage()
    for model_name in model_names:
        path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, model_name))
        name = 'R2_group_fdr_effect'
        maps.append(fetch_map(path, name)[0])
    
    arrays = [vol_to_surf(map_, fsaverage[kwargs['surf_mesh']]) for map_ in maps]

    data_tmp = np.vstack(arrays)
    img = np.argmax(data_tmp, axis=0)
    return img

def prepare_data_for_anova(
    model_names, 
    atlas_maps, 
    labels, 
    global_mask,
    object_of_interest='R2', 
    language='english', 
    includ_context=False,
    includ_norm=False,
    OUTPUT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english'
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

    for index_mask in tqdm(range(len(labels)-1)):
        masker = get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask)
        for name in data.keys():
            for subject in data[name].keys():
                try:
                    array = masker.transform(data[name][subject][object_of_interest])
                    mean = np.mean(array)
                    median = np.median(array)
                    third_quartile = np.percentile(array, 75)
                except ValueError:
                    mean = np.nan
                    median = np.nan
                    third_quartile = np.nan
                result.append([name, subject, labels[index_mask + 1], mean, median, third_quartile])
    result = pd.DataFrame(result, columns=['model', 'subject', 'ROI', 'R2_mean', 'R2_median', 'R2_3rd_quartile'])
    #if includ_context or includ_norm:
    #    final_result = []
    #    for index, row in result.iterrows():
    #        context_pre = row['model'].split('pre-')[1].split('_')[0] if includ_context else None
    #        context_post = row['model'].split('_norm_')[0].split('-')[-1] if includ_context else None
    #        norm = row['model'].split('_norm_')[1].split('_')[0] if includ_norm else None
    #        name = row['model'].split('_pre-')[0] + '_'.join(row['model'].split('_')[-3:])
    #        final_result.append([name, context_pre, context_post, norm, row['subject'], row['ROI'], row['R2_mean'], row['R2_median'], row['R2_3rd_quartile']])
    #    result = pd.DataFrame(final_result, columns=['model', 'context_pre', 'context_post', 'norm', 'subject', 'ROI', 'R2_mean', 'R2_median', 'R2_3rd_quartile'])
    #print("\t\t-->Done")
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

def aggregate_beta_maps(data, nb_layers=13, layer_size=768):
    """Clustering of voxels or ROI based on their beta maps.
    maps: (#voxels x #features)
    """
    if isinstance(data, str):
        data = np.load(data)
    nb_layers = nb_layers
    layer_size = layer_size
    result = np.zeros((data.shape[0], nb_layers))
    for index in range(nb_layers):
        result[:, index] = np.mean(data[:, layer_size * index : layer_size * (index + 1)], axis=1)
    return result

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
    surnames, 
    legend_names, 
    syntactic_roi, 
    language_roi, 
    figsize=(9,12), 
    count=False, 
    title=None, 
    ylabel='Regions of interest (ROI)', 
    xlabel='R2 value', 
    model_name=''
    ):
    """Plots models vertically.
    """
    #limit = (-0.01, 0.05) if object_of_interest=='R2' else None
    limit = None
    dash_inf = limit[0] if limit is not None else 0
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
    plt.xlim(limit)
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
    legend_names, 
    analysis_name, 
    save_folder, 
    label_names,
    figsize=(9,20), 
    title=None, 
    ylabel='Regions of interest (ROI)', 
    xlabel='R2 value', 
    model_name=''
    ):
    """Plots models horizontally.
    """

    limit = None
    dash_inf = limit[0] if limit is not None else 0
    plt.figure(figsize=figsize) # (7.6,12)
    ax = plt.axes()
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1, data.shape[0] + 1)]
    for col in range(data.shape[0]):
        plt.plot(legend_names, data[col, :], '.-', alpha=0.7, markersize=9, color=colors[col])
    plt.title(title)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

    plt.xlim(limit)
    plt.minorticks_on()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='y')
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='x')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='x')
    plt.legend(label_names, ncol=3, bbox_to_anchor=(0,0,1,1), fontsize=10)

    plt.tight_layout()
    check_folder(save_folder)
    if save_folder:
        plt.savefig(os.path.join(save_folder, '{model_name}-{analysis_name}.png'.format(model_name=model_name,
                                                                                        analysis_name=analysis_name)))
        plt.close('all')
    else:
        plt.show()

def plot_roi_img_surf(surf_img, saving_path, plot_name, inflated=False, compute_surf=True, colorbar=True, **kwargs):
    fsaverage = datasets.fetch_surf_fsaverage()
    if compute_surf:
        surf_img = vol_to_surf(surf_img, fsaverage[kwargs['surf_mesh']])
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
        colorbar=colorbar)
    if saving_path:
        disp.savefig(saving_path + plot_name + '_{}_{}_{}.png'.format(kwargs['surf_mesh_type'], kwargs['hemi'], kwargs['view']))
    else:
        plotting.show()

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

def plot_img_surf(surf_img, saving_path, plot_name, inflated=False, compute_surf=True, **kwargs):
    fsaverage = datasets.fetch_surf_fsaverage()
    if compute_surf:
        surf_img = vol_to_surf(surf_img, fsaverage[kwargs['surf_mesh']])
    if inflated:
        kwargs['surf_mesh'] = 'infl_left' if 'left' in kwargs['surf_mesh_type'] else 'infl_right' 
    disp = plotting.plot_surf_stat_map(
                        surf_mesh=fsaverage[kwargs['surf_mesh']], 
                        stat_map=surf_img,
                        hemi=kwargs['hemi'], 
                        view=kwargs['view'],
                        bg_map=fsaverage[kwargs['bg_map']], 
                        bg_on_data=kwargs['bg_on_data'],
                        darkness=kwargs['darkness'])
    if saving_path:
        disp.savefig(saving_path + plot_name + '_{}_{}_{}.png'.format(kwargs['surf_mesh_type'], kwargs['hemi'], kwargs['view']))
    else:
        plotting.show()

def clustering(path_to_beta_maps, clustering_method, global_mask=None, atlas_maps=None, labels=None, **kwargs):
    """Clustering of voxels or ROI based on their beta maps.
    """
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps).T
    else:
        data = path_to_beta_maps.T
    if global_mask and atlas_maps:
        global_masker = load_masker(global_mask)
        imgs = global_masker.inverse_transform([data[i, :] for i in range(data.shape[0])])
        data = np.zeros((data.shape[0], len(labels)-1))
        for index_mask in tqdm(range(len(labels)-1)):
            masker = get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask)
            data[:, index_mask] = np.mean(masker.transform(imgs), axis=1)
    data = data.T
    if clustering_method=="umap":
        clusterable_embedding = umap.UMAP(**kwargs).fit_transform(data)
        # n_neighbors=30,
        # min_dist=0.0,
        # n_components=10,
        # random_state=SEED,
        labels = hdbscan.HDBSCAN(**kwargs).fit_predict(clusterable_embedding)
        # min_samples=10,
        # min_cluster_size=500,
                    
        ##clustered = (labels >= 0)
        ##standard_embedding = umap.UMAP(random_state=42).fit_transform(data)
        ##plt.scatter(standard_embedding[~clustered, 0],
        ##            standard_embedding[~clustered, 1],
        ##            c=(0.5, 0.5, 0.5),
        ##            s=0.1,
        ##            alpha=0.5)
        ##plt.scatter(standard_embedding[clustered, 0],
        ##            standard_embedding[clustered, 1],
        ##            c=labels[clustered],
        ##            s=0.1,
        ##            cmap='Spectral')
    elif clustering_method=="max":
        labels = np.argmax(data, axis=1)
    else:
        sc = AgglomerativeClustering(**kwargs) # n_clusters=n_clusters, linkage=“ward”, “complete”, “average”, “single”
        clustering = sc.fit(data)
        labels = clustering.labels_
    return labels


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
        title=None, 
        vmax=vmax)
    if output_dir:
        nib.save(eff_map, os.path.join(output_dir, "{}_group_effect.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_group_effect".format(name)))
    else:
        plotting.show()
    #thr = thresholded_zmap.get_data()
    thr = np.abs(z_map.get_data())
    eff = eff_map.get_data()
    thr_eff = eff * (thr > 3.09) # ? z_th ?
    eff_thr_map = new_img_like(eff_map, thr_eff)
    display = plotting.plot_glass_brain(
        smooth_img(eff_thr_map, fwhm=fwhm),
        colorbar=True,
        plot_abs=False,
        display_mode='lzry',
        title=None,
        vmax=vmax)
    if output_dir:
        nib.save(eff_thr_map, os.path.join(output_dir, "{}_group_fdr_effect.nii.gz".format(name)))
        display.savefig(os.path.join(output_dir, "{}_group_fdr_effect".format(name)))
    else:
        plotting.show()
    plt.close('all')

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
    observed_data='R2',
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
    name = '{}-vs-{}_{}'.format(model1_name, model2_name, analysis_name)
    output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
    check_folder(output_dir)
    create_one_sample_t_test(name + '_' + observed_data, imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=None)


def compute_t_test_for_model_comparison(
    data, 
    smoothing_fwhm=None, 
    language='english',
    vmax=None,
    PROJECT_PATH='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    ):
    for name in data.keys():
        # R2
        name = name.replace('_{}', '')
        imgs = data[name]['R2']
        output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format(language, name))
        check_folder(output_dir)
        create_one_sample_t_test(name + '_R2', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=vmax)
        # Pearson coefficient
        imgs = data[name]['Pearson_coeff']
        create_one_sample_t_test(name + '_Pearson_coeff', imgs, output_dir, smoothing_fwhm=smoothing_fwhm, vmax=vmax)

def anova(data, anova_type, dv, within=None, between=None, subject=None, detailed=True, correction=False, effsize='np2', method='pearson', padjust='fdr_bh', columns=None):
    """Compute various ANOVA type analysis on data. Available functions are:
    anova, rm_anova, pairwise_ttests, mixed_anova, pairwise_corr.
    """
    if anova_type=='anova':
        result = pg.anova(data=data, dv=dv, between=between, detailed=detailed)
    elif anova_type=='rm_anova':
        result = pg.rm_anova(data=data, dv=dv, within=within, subject=subject, detailed=detailed)
    elif anova_type=='pairwise_ttests':
        result = pg.pairwise_ttests(data=data, dv=dv, within=within, subject=subject,
                             parametric=True, padjust=padjust, effsize='hedges')
    elif anova_type=='mixed_anova':
        result = pg.mixed_anova(data=data, dv=dv, between=between, within=within,
                     subject=subject, correction=False, effsize=effsize)
    elif anova_type=='pairwise_corr':
        result = pg.pairwise_corr(data, columns=columns, method=method)
    pg.print_table(result, floatfmt='.3f')
    return result

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
