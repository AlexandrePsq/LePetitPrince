import sys
import os

from itertools import combinations, product
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker, MultiNiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_glass_brain, plot_img
import nibabel as nib
from nilearn.regions import RegionExtractor

from joblib import Parallel, delayed
import yaml
import pandas as pd
import argparse
import glob
from textwrap import wrap 

import warnings
warnings.simplefilter(action='ignore')

local = False
root = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince' if local else '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
paths = {'path2root': root}
paths['path2derivatives'] = os.path.join(paths['path2root'], 'derivatives')
paths['path2data'] = os.path.join(paths['path2root'], 'data')

def check_folder(path):
    """Create the adequate folders for
    the path to exist.
    """
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def fetch_ridge_maps(model, subject, value):
    """Retrieve the R2/pearson values (significant or not)
    for a given subject and model.
    """
    path = os.path.join(paths['path2derivatives'], 'fMRI', 'ridge-indiv', 'english', subject, model, 'outputs', 'maps', '*{}*.nii.gz'.format(value))
    files = sorted(glob.glob(path))
    return files[0]

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

def compute_global_masker(files, smoothing_fwhm=None): # [[path, path2], [path3, path4]]
    """Return a MultiNiftiMasker object.
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, smoothing_fwhm=smoothing_fwhm) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def save_hist(distribution, path):
    plt.hist(distribution[~np.isnan(distribution)], bins=50)
    plt.savefig(path)
    plt.close()

def process_matrix(subjects, model1, model2, path2data1, path2data2, compute_significant=False):
    print("Processing subject for model 1: {} and model 2: {}...".format(model1, model2))
    name = surnames[model1] + '-' + surnames[model2]
    print("\tLoading r2 values...")
    path1 = os.path.join(path2data1,'{subject}/{model}', 'outputs', 'r2.npy')
    path2 = os.path.join(path2data2,'{subject}/{model}', 'outputs', 'r2.npy')
    data1 = [np.load(path1.format(subject=subject, model=model1)) for subject in subjects]
    data2 = [np.load(path2.format(subject=subject, model=model2)) for subject in subjects]
    print("\t\t-->Done")
    print("\tDifferenciating...")
    data = np.mean(np.vstack([data1[index] - array for index, array in enumerate(data2)]), axis=0)
    print("\t\t-->Done")
    print("\tSmoothing...")
    data_smoothed = np.mean(np.vstack([smooth3D(data1[index], global_masker) - smooth3D(array, global_masker) for index, array in enumerate(data2)]), axis=0)
    print("\t\t-->Done")
    if compute_significant:
        print("\tLoading distributions...")
        path1 = os.path.join(path2data1,'{subject}/{model}', 'outputs', 'distribution_r2.npy')
        path2 = os.path.join(path2data2,'{subject}/{model}', 'outputs', 'distribution_r2.npy')
        distribution1 = [np.mean(np.load(path1.format(subject=subject, model=model1)), axis=0) for subject in subjects]
        distribution2 = [np.mean(np.load(path2.format(subject=subject, model=model2)), axis=0) for subject in subjects]
        print("\t\t-->Done")
        print("\tDifferenciating distributions...")
        distribution = [array - distribution2[index] for index, array in enumerate(distribution1)]
        distribution = np.mean(distribution, axis=0)
        print("\t\t-->Done")
        print("\tSmoothing distributions...")
        distribution1 = [np.apply_along_axis(lambda x: smooth3D(x, global_masker), 1, array).reshape(array.shape) for array in distribution1]
        distribution2 = [np.apply_along_axis(lambda x: smooth3D(x, global_masker), 1, array).reshape(array.shape) for array in distribution2]
        diff = [distribution1[index] - distribution2[index] for index, _ in enumerate(distribution1)]
        distribution_smoothed = np.mean(diff, axis=0)
        print("\t\t-->Done")
        print("\tComputing p values...")
        p_values = (1.0 * np.sum(distribution>data, axis=0))/distribution.shape[0] 
        p_values_smoothed = (1.0 * np.sum(distribution_smoothed>data_smoothed, axis=0))/distribution_smoothed.shape[0]
        print("\t\t-->Done")
        print("\tComputing mask...")
        mask = (p_values < (1-99.9/100))
        mask_smoothed = (p_values_smoothed < (1-99.9/100))
        print("\t\t-->Done")
        print("\tComputing significant values...")
        significant = np.zeros(data.shape)
        significant[mask] = data[mask]
        significant[~mask] = np.nan
        significant_smoothed = np.zeros(data_smoothed.shape)
        significant_smoothed[mask_smoothed] = data_smoothed[mask_smoothed]
        significant_smoothed[~mask_smoothed] = np.nan
        print("\t\t-->Done")
        print("\tPlotting and saving...")
        save_glass_brain(significant, name, source='fMRI', language='english')
        save_glass_brain(significant_smoothed, name + '-smoothed', source='fMRI', language='english')
    else:
        print("\tPlotting and saving...")
        save_glass_brain(data, name, source='fMRI', language='english')
    print("\t-->Done")

def save_glass_brain(distribution, name, source='fMRI', language='english'):
    #img = global_masker.inverse_transform(distribution)
    #display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=False, plot_abs=False)
    save_folder = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', "Appendix")
    check_folder(save_folder)
    np.save(os.path.join(save_folder, name + '.npy'), distribution)
    #nib.save(img, os.path.join(save_folder, name + '.nii.gz'))
    #display.savefig(os.path.join(save_folder, name + '.png'))
    #display.close()
    print("\t\t-->Done")

def smooth3D(vector, masker):
    img = smooth_img(masker.inverse_transform(vector), fwhm=6)
    return masker.transform(img)[0]

def vertical_plot(data, x_names, analysis_name, save_folder, object_of_interest, surnames, models, syntactic_roi, language_roi, figsize=(7.6,12), count=False, title=None, ylabel='Regions of interest (ROI)', xlabel='R2 value'):
    """Plots vertically the 6 first models.
    """
    limit = (-0.02, 0.05) if object_of_interest=='maps_r2_' else None
    dash_inf = limit[0] if limit else 0
    x = x_names.copy()
    plt.figure(figsize=figsize)
    ax = plt.axes()
    order = np.argsort(np.mean(data, axis=1))
    data = data[order, :]
    x = [x[i] for i in order]
    ax = plt.axes()
    plt.plot(data[:,0], x, '.k', alpha=0.7, markersize=12)
    plt.plot(data[:,1], x, '.g', alpha=0.7, markersize=12)
    plt.plot(data[:,2], x, '.r', alpha=0.7, markersize=12)
    plt.plot(data[:,3], x, '.b', alpha=0.5, markersize=12)
    plt.plot(data[:,4], x, '^r', alpha=0.3, markersize=8)
    plt.plot(data[:,5], x, '^b', alpha=0.3, markersize=8)
    #plt.title(title)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    if count:
        plt.axvline(x=100, color='k', linestyle='-', alpha=0.2)
    plt.hlines(x, dash_inf, data[:,0], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,1], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,2], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,3], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,4], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,5], linestyle="dashed", alpha=0.1)
    plt.xlim(limit)
    plt.minorticks_on()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='x')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='x')
    #plt.legend(plot, [surnames[model] for model in models], ncol=3, bbox_to_anchor=(0,0,1,1), fontsize=5)
    for index, label in enumerate(x):
        if label in language_roi:
            if label in syntactic_roi:
                ax.get_yticklabels()[index].set_bbox(dict(facecolor="green", alpha=0.4)) # set box around label
            #ax.axhspan(index-0.5, index+0.5, alpha=0.1, color='red') #set shade over the line of interest
    plt.tight_layout()
    check_folder(save_folder)
    plt.savefig(os.path.join(save_folder, '{analysis_name}.png'.format(analysis_name=analysis_name)))
    plt.close()

def layer_plot(X,Y, error, surnames, object_of_interest, roi, limit, save_folder, name):
    """Layer-wised plot per ROI
    """
    plot = plt.plot(X, Y)
    ax = plt.axes()
    plt.minorticks_on()
    plt.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='both')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='y')
    #plt.title('\n'.join(wrap(surnames[object_of_interest] + ' per ROI - {roi} - all voxels'.format(roi=roi))))
    plt.xlabel('Models', fontsize=16)
    plt.ylabel(surnames[object_of_interest], fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    plt.ylim(0,0.1)
    plt.xticks(rotation=30, fontsize=10, horizontalalignment='right')
    plt.errorbar(X, Y, error, linestyle='None', marker='^')
    plt.tight_layout()
    check_folder(save_folder)
    plt.savefig(os.path.join(save_folder, name))
    plt.close()

all_models = ['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer',
                'bert_bucket_pca_300_all-layers',
                'gpt2_pca_300_all-layers',
                'bert_bucket_all-layers',
                'gpt2_all-layers',
                'gpt2_embeddings',
                'gpt2_layer-1',
                'gpt2_layer-2',
                'gpt2_layer-3',
                'gpt2_layer-4',
                'gpt2_layer-5',
                'gpt2_layer-6',
                'gpt2_layer-7',
                'gpt2_layer-8',
                'gpt2_layer-9',
                'gpt2_layer-10',
                'gpt2_layer-11',
                'gpt2_layer-12',
                'bert_bucket_embeddings',
                'bert_bucket_layer-1',
                'bert_bucket_layer-2',
                'bert_bucket_layer-3',
                'bert_bucket_layer-4',
                'bert_bucket_layer-5',
                'bert_bucket_layer-6',
                'bert_bucket_layer-7',
                'bert_bucket_layer-8',
                'bert_bucket_layer-9',
                'bert_bucket_layer-10',
                'bert_bucket_layer-11',
                'bert_bucket_layer-12',
                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer']

limit_values = {'maps_r2_':0.07,
                'maps_pearson_corr':0.6,
                'maps_significant_pearson_corr_with_pvalues':0.4,
                'maps_significant_r2_with_pvalues':0.07}

syntactic_roi = ['Inferior Frontal Gyrus, pars triangularis',
                    'Inferior Frontal Gyrus, pars opercularis',
                    'Temporal Pole',
                    'Superior Temporal Gyrus, anterior division',
                    'Superior Temporal Gyrus, posterior division',
                    'Middle Temporal Gyrus, anterior division',
                    'Middle Temporal Gyrus, posterior division',
                    'Middle Temporal Gyrus, temporooccipital part',
                    'Angular Gyrus']

language_roi = ['Inferior Frontal Gyrus, pars triangularis',
                    'Inferior Frontal Gyrus, pars opercularis',
                    'Temporal Pole',
                    'Superior Temporal Gyrus, anterior division',
                    'Superior Temporal Gyrus, posterior division',
                    'Middle Temporal Gyrus, anterior division',
                    'Middle Temporal Gyrus, posterior division',
                    'Middle Temporal Gyrus, temporooccipital part',
                    'Angular Gyrus']

surnames = {'wordrate_word_position': 'Word-position',
                'wordrate_model': 'Wordrate',
                'wordrate_log_word_freq': 'Log-word-frequency',
                'wordrate_function-word': 'Function-words',
               'wordrate_content-word': 'Content-words',
                'wordrate_all_model': 'All-word-related-models',
                'topdown_model': 'Topdown-parser',
                'rms_model': 'RMS',
                'other_sentence_onset': 'Sentence-onset',
                'mfcc_model': 'MFCC',
                'gpt2_layer-9': 'GPT2-L9',
                'gpt2_layer-8': 'GPT2-L8',
                'gpt2_layer-7': 'GPT2-L7',
                'gpt2_layer-6': 'GPT2-L6',
                'gpt2_layer-5': 'GPT2-L5',
                'gpt2_layer-4': 'GPT2-L4',
                'gpt2_layer-3': 'GPT2-L3',
                'gpt2_layer-2': 'GPT2-L2',
                'gpt2_layer-12': 'GPT2-L12',
                'gpt2_layer-11': 'GPT2-L11',
                'gpt2_layer-10': 'GPT2-L10',
                'gpt2_layer-1': 'GPT2-L1',
                'gpt2_embeddings': 'GPT2-L0',
                'gpt2_all-layers': 'GPT2-all',
                'bottomup_model': 'Bottomup-parser',
                'bert_bucket_median_all-layers': 'BERT-median-all',
                'bert_bucket_layer-9': 'BERT-L9',
                'bert_bucket_layer-8': 'BERT-L8',
                'bert_bucket_layer-7': 'BERT-L7',
                'bert_bucket_layer-6': 'BERT-L6',
                'bert_bucket_layer-5': 'BERT-L5',
                'bert_bucket_layer-4': 'BERT-L4',
                'bert_bucket_layer-3': 'BERT-L3',
                'bert_bucket_layer-2': 'BERT-L2',
                'bert_bucket_layer-12': 'BERT-L12',
                'bert_bucket_layer-11': 'BERT-L11',
                'bert_bucket_layer-10': 'BERT-L10',
                'bert_bucket_layer-1': 'BERT-L1',
                'bert_bucket_embeddings': 'BERT-L0',
                'bert_bucket_all-layers': 'BERT-all',
                'glove_embeddings': 'GloVe',
                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer': 'LSTM-E600-H768-#L1-all',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer': 'LSTM-E600-H300-#L1-all',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_all-layers': 'LSTM-E600-H150-#L2-all',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_all-layers': 'LSTM-E600-H100-#L3-all',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_all-layers': 'LSTM-E600-H75-#L4-all',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_first-layer': 'LSTM-E600-H150-#L2-L1',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_second-layer': 'LSTM-E600-H150-#L2-L2',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer': 'LSTM-E600-H100-#L3-L1',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer': 'LSTM-E600-H100-#L3-L2',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer': 'LSTM-E600-H100-#L3-L3',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer': 'LSTM-E600-H75-#L4-L1',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer': 'LSTM-E600-H75-#L4-L2',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer': 'LSTM-E600-H75-#L4-L3',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer': 'LSTM-E600-H75-#L4-L4',
                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer': 'LSTM-E600-C768-#L1-all',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_cell_first-layer': 'LSTM-E600-C300-#L1-all',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_all-layers': 'LSTM-E600-C150-#L2-all',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_all-layers': 'LSTM-E600-C100-#L3-all',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_all-layers': 'LSTM-E600-C75-#L4-all',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_first-layer': 'LSTM-E600-C150-#L2-L1',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_second-layer': 'LSTM-E600-C150-#L2-L2',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_first-layer': 'LSTM-E600-C100-#L3-L1',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_second-layer': 'LSTM-E600-C100-#L3-L2',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_third-layer': 'LSTM-E600-C100-#L3-L3',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_first-layer': 'LSTM-E600-C75-#L4-L1',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_second-layer': 'LSTM-E600-C75-#L4-L2',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_third-layer': 'LSTM-E600-C75-#L4-L3',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_fourth-layer': 'LSTM-E600-C75-#L4-L4',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_775': 'LSTM-E650-H650-Cu775',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_775_987': 'LSTM-E650-H650-Cu775-987',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_987': 'LSTM-E650-H650-Cu987',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_775': 'LSTM-E650-H650-Fu775',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_987': 'LSTM-E650-H650-Fu775',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_short-range-units': 'LSTM-E650-H650-Short-Range',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_unit_1149': 'LSTM-E650-H650-Hu1149',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_input_unit_775': 'LSTM-E650-H650-Iu775',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_input_unit_987': 'LSTM-E650-H650-Iu987',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_775_987': 'LSTM-E650-H650-Fu775-987',
                'lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_unit_1282_1283': 'LSTM-E650-H650-Hu1282-1283',
                'bert_bucket_pca_300_all-layers': 'BERT-all-pca-300',
                'gpt2_pca_300_all-layers':'GPT2-all-pca-300',
                'maps_r2_':'R2', 
                'maps_pearson_corr':'Pearson',
                'maps_significant_pearson_corr_with_pvalues':'Significant-Pearson', 
                'maps_significant_r2_with_pvalues':'Significant-R2',
                'Background': 'Background', 
                'Frontal Pole': 'FP', 
                'Insular Cortex': 'InsC', 
                'Superior Frontal Gyrus': 'SFG', 
                'Middle Frontal Gyrus': 'MFG', 
                'Inferior Frontal Gyrus, pars triangularis': 'IFG, pars tr.', 
                'Inferior Frontal Gyrus, pars opercularis': 'IFG, pars op.', 
                'Precentral Gyrus': 'PrG', 
                'Temporal Pole': 'TP', 
                'Superior Temporal Gyrus, anterior division': 'STG-a', 
                'Superior Temporal Gyrus, posterior division': 'STG-p', 
                'Middle Temporal Gyrus, anterior division': 'MTG-a', 
                'Middle Temporal Gyrus, posterior division': 'MTG-p', 
                'Middle Temporal Gyrus, temporooccipital part': 'MTG-toc', 
                'Inferior Temporal Gyrus, anterior division': 'ITG-a', 
                'Inferior Temporal Gyrus, posterior division': 'ITG-p', 
                'Inferior Temporal Gyrus, temporooccipital part': 'ITG-toc', 
                'Postcentral Gyrus': 'PoG', 
                'Superior Parietal Lobule': 'SPL', 
                'Supramarginal Gyrus, anterior division': 'SmG-a', 
                'Supramarginal Gyrus, posterior division': 'SmG-p', 
                'Angular Gyrus': 'AnG', 
                'Lateral Occipital Cortex, superior division': 'LOC-s', 
                'Lateral Occipital Cortex, inferior division': 'LOC-i', 
                'Intracalcarine Cortex': 'IccC', 
                'Frontal Medial Cortex': 'FMC', 
                'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'SMA', 
                'Subcallosal Cortex': 'SC', 
                'Paracingulate Gyrus': 'PCgG', 
                'Cingulate Gyrus, anterior division': 'CgG-a', 
                'Cingulate Gyrus, posterior division': 'CgG-p', 
                'Precuneous Cortex': 'PrCuG', 
                'Cuneal Cortex': 'CuC', 
                'Frontal Orbital Cortex': 'FoG', 
                'Parahippocampal Gyrus, anterior division': 'PHG-a', 
                'Parahippocampal Gyrus, posterior division': 'PHG-p', 
                'Lingual Gyrus': 'LgG', 
                'Temporal Fusiform Cortex, anterior division': 'TFuC-a', 
                'Temporal Fusiform Cortex, posterior division': 'TFuC-p', 
                'Temporal Occipital Fusiform Cortex': 'TocFucC', 
                'Occipital Fusiform Gyrus': 'OcFuC', 
                'Frontal Operculum Cortex': 'FOpC', 
                'Central Opercular Cortex': 'cOpC', 
                'Parietal Operculum Cortex': 'POpC', 
                'Planum Polare': 'PP', 
                "Heschl's Gyrus (includes H1 and H2)": 'HG', 
                'Planum Temporale': 'PT', 
                'Supracalcarine Cortex': 'SccC', 
                'Occipital Pole': 'OcP'}

syntactic_roi += [surnames[item] for item in syntactic_roi]
language_roi += [surnames[item] for item in language_roi]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--model1", type=str)
    parser.add_argument("--model2", type=str)
    parser.add_argument("--path2data1", type=str)
    parser.add_argument("--path2data2", type=str)

    args = parser.parse_args()

    paths2check = []
    source = 'fMRI'
    language = 'english'
    #subjects = ['sub-057', 'sub-063', 'sub-067', 'sub-073', 'sub-077', 'sub-082', 'sub-101', 'sub-109', 'sub-110', 'sub-113', 'sub-114']
    subjects = ['sub-057', 'sub-058', 'sub-059', 'sub-061', 'sub-062', 'sub-063', 'sub-064', 'sub-065', 
            'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-072', 'sub-073', 'sub-074', 
            'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081', 'sub-082', 
            'sub-083', 'sub-084', 'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-091', 'sub-092', 
            'sub-093', 'sub-094', 'sub-095', 'sub-096', 'sub-097', 'sub-098', 'sub-099', 'sub-100', 
            'sub-101', 'sub-103', 'sub-104', 'sub-105', 'sub-106', 'sub-108', 'sub-109', 'sub-110', 
            'sub-113', 'sub-114', 'sub-115']

    # Sanity check
    for path in paths2check:
        check_folder(path)

    # retrieve default atlas  (= set of ROI)
    atlas = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
    labels = atlas['labels']
    labels = [surnames[value] for value in labels]
    maps = nilearn.image.load_img(atlas['maps'])

    # Compute global masker
    fmri_runs = {}
    print("Computing global masker...")
    for subject in subjects:
        fmri_path = os.path.join(paths['path2root'], "data/fMRI/{language}/{subject}/func/")
        check_folder(fmri_path)
        fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path.format(language=language, subject=subject), 'fMRI_*run*')))
    global_masker = compute_global_masker(list(fmri_runs.values()))
    #global_masker_smoothed = compute_global_masker(list(fmri_runs.values()), smoothing_fwhm=5)
    print("\t-->Done")


    ###########################################################################
    #################### 1 (Individual features) ####################
    ###########################################################################
    print("Computing: 1 (Individual features)")
    #plots = [['wordrate_model'],
    #               ['wordrate_log_word_freq'],
    #               ['mfcc_model'],
    #               ['bert_bucket_all-layers'],
    #               ['bert_bucket_embeddings'],
    #               ['bert_bucket_layer-1'],
    #               ['bert_bucket_layer-2'],
    #               ['bert_bucket_layer-3'],
    #               ['bert_bucket_layer-4'],
    #               ['bert_bucket_layer-5'],
    #               ['bert_bucket_layer-6'],
    #               ['bert_bucket_layer-7'],
    #               ['bert_bucket_layer-8'],
    #               ['bert_bucket_layer-9'],
    #               ['bert_bucket_layer-10'],
    #               ['bert_bucket_layer-11'],
    #               ['bert_bucket_layer-12'],
    #               ['gpt2_all-layers'],
    #               ['gpt2_embeddings'],
    #               ['gpt2_layer-1'],
    #               ['gpt2_layer-2'],
    #               ['gpt2_layer-3'],
    #               ['gpt2_layer-4'],
    #               ['gpt2_layer-5'],
    #               ['gpt2_layer-6'],
    #               ['gpt2_layer-7'],
    #               ['gpt2_layer-8'],
    #               ['gpt2_layer-9'],
    #               ['gpt2_layer-10'],
    #               ['gpt2_layer-11'],
    #               ['gpt2_layer-12'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer'],
    #                ['glove_embeddings'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_all-layers'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_all-layers'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_all-layers'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_in_all-layers'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_out_all-layers'],
    #                ['bert_bucket_pca_300_all-layers'],
    #                ['gpt2_pca_300_all-layers'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_775'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_775_987'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_cell_unit_987'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_775'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_987'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_short-range-units'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_unit_1149'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_input_unit_775'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_input_unit_987'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_0'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_0_650'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_forget_unit_775_987'],
    #                ['lstm_wikikristina_embedding-size_650_nhid_650_nlayers_2_dropout_02_hidden_unit_1282_1283']]
    #plots= [['rms_model']]
    #for object_of_interest in ['maps_r2_', 'maps_significant_r2_with_pvalues']:
    #    for models in plots:
    #        all_paths = []
    #        for subject in subjects:
    #            model_name = models[0]
    #            path_template = os.path.join(paths['path2root'], "derivatives/fMRI/ridge-indiv/{}/{}/{}/outputs/maps/*{}*.nii.gz".format(language, subject, model_name, object_of_interest))
    #            path = sorted(glob.glob(path_template))[0]
    #            all_paths.append(path)
    #            img = load_img(path) 
    #            display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=False, plot_abs=False)
    #            save_folder = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', "1", model_name, surnames[object_of_interest], subject)
    #            check_folder(save_folder)
    #            display.savefig(os.path.join(save_folder, model_name + '-' + surnames[object_of_interest] + '-' + subject  + '.png'))
    #            display.close()
    #        #data = [global_masker.transform(path) for path in all_paths]
    #        #for index, array in enumerate(data):
    #        #    array[array<0] = 0
    #        #data = np.vstack(data)
    #        #data = np.mean(data, axis=0)
    #        #img = global_masker.inverse_transform(data)
    #        img = mean_img([threshold_img(fetch_ridge_maps(model_name, subject, object_of_interest), threshold=0) for subject in subjects])
    #        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=False, plot_abs=False, vmax=None) #0.075
    #        save_folder = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', "1", model_name, surnames[object_of_interest])
    #        check_folder(save_folder)
    #        display.savefig(os.path.join(save_folder, model_name + '-' + surnames[object_of_interest] + '-' + '-averaged-'  + '.png'))
    #        display.close()
#
    #print("\t\t-->Done")

    ############################################################################
    ################ 2 (Comparison 300 features models) ###############
    ############################################################################
    print("Computing: 2 (Comparison 300 features models)...")
    models = ['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
                'bert_bucket_pca_300_all-layers',
                'gpt2_pca_300_all-layers']
    #            'bert_bucket_all-layers',
     #           'gpt2_all-layers']
    plot_name = ["Comparison 300 features models"]
    data = {model:[] for model in all_models}
    for value in ['maps_r2_']:
        tmp_path = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', "2")
        check_folder(os.path.join(tmp_path, 'averages'))

        print("\tTransforming data...")
        for model_index, model in enumerate(models):
            data[model] = [global_masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
            distribution = np.mean(np.vstack(data[model]), axis=0)
            save_hist(distribution, os.path.join(tmp_path, 'averages', 'hist_averaged_{}_{}.png'.format(value, model)))
            path2output_raw = os.path.join(paths['path2derivatives'], 'fMRI/ridge-indiv', language, 'averaged', model)
            check_folder(path2output_raw)
            nib.save(global_masker.inverse_transform(distribution), os.path.join(path2output_raw, value + '.nii.gz'))
        print("\t\t-->Done")

        x_labels = labels[1:]
        mean = np.zeros((len(labels)-1, len(models)))
        mean_filtered = np.zeros((len(labels)-1, len(models)))
        # extract data
        print("\tLooping through labeled masks...")
        for index_mask in range(len(labels)-1):
            mask = math_img('img > 50', img=index_img(maps, index_mask))  
            masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
            masker.fit()

            for index_model, model in enumerate(models):
                path2output_raw = os.path.join(paths['path2derivatives'], 'fMRI/ridge-indiv', language, 'averaged', model)
                array = masker.transform(os.path.join(path2output_raw, value + '.nii.gz'))
                filtered_data = [masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
                mean_filtered[index_mask, index_model] = np.mean(np.array([np.mean(array[array>np.percentile(array, 75)]) for array in filtered_data]))
                mean[index_mask, index_model] = np.mean(array)
        print("\t\t-->Done")

        print("\tPlotting...")
        # save plots
        vertical_plot(mean, x_labels, 'Mean-comparison-300-features', 
                        os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi, xlabel='R2 values')
        vertical_plot(mean_filtered, x_labels, 'Mean-filtered-comparison-300-features', 
                        os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '2'), 
                        'filtered', surnames, models, syntactic_roi, language_roi, xlabel='Filtered R2 values')
        print("\t\t-->Done")

    ###########################################################################
    ############### 2 bis (avec significant R2 values) ###############
    ###########################################################################
    #print("Dealing with significant R2...")
    #value = 'maps_significant_r2_with_pvalues'
    #counter = np.zeros((len(labels)-1, len(models)))
    #mean = np.zeros((len(labels)-1, len(models)))
    #maximum = np.zeros((len(labels)-1, len(models)))
    #x_labels = labels[1:]
    #for index_mask in range(len(labels)-1):
    #    mask = math_img('img > 50', img=index_img(maps, index_mask))  
    #    masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
    #    masker.fit()
    #    for index_model, model in enumerate(models):
    #        data0 = [masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
    #        data = []
    #        for array in data0:
    #            if np.count_nonzero(array)>0:
    #                data.append(array)
    #        counter[index_mask, index_model] = np.mean([np.count_nonzero(array) for array in data])
    #        mean[index_mask, index_model] = np.mean([np.mean(array[array!=0]) for array in data])
    #        maximum[index_mask, index_model] = np.mean([np.max(array[array!=0]) for array in data])
    #vertical_plot(counter, x_labels, 'Count-non-Nan-values', 
    #                    os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '2-significative'), 
    #                    value, surnames, models, syntactic_roi, language_roi, count=True, 
    #                    ylabel='Regions of interest (ROI)', xlabel='Count')
    #vertical_plot(mean, x_labels, 'Mean-significant-R2', 
    #                    os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '2-significative'), 
    #                    value, surnames, models, syntactic_roi, language_roi)
    #vertical_plot(maximum, x_labels, 'Maximum-significant-R2', 
    #                    os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '2-significative'), 
    #                    value, surnames, models, syntactic_roi, language_roi)
    #print("\t-->Done")


    ###########################################################################
    ############### 3 (Layer-wise analysis) ###############
    ###########################################################################
    #print("Computing: 3 (Layer-wise analysis)...")
    #plots = {
    #    "LSTM-#L3":['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer'],
    #    "LSTM-#L4":['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer']
    #}
    #plots = {"GPT2":['gpt2_embeddings',
    #                'gpt2_layer-1',
    #                'gpt2_layer-2',
    #                'gpt2_layer-3',
    #                'gpt2_layer-4',
    #                'gpt2_layer-5',
    #                'gpt2_layer-6',
    #                'gpt2_layer-7',
    #                'gpt2_layer-8',
    #                'gpt2_layer-9',
    #                'gpt2_layer-10',
    #                'gpt2_layer-11',
    #                'gpt2_layer-12',
    #                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer'],
    #        "BERT":['bert_bucket_embeddings',
    #                'bert_bucket_layer-1',
    #                'bert_bucket_layer-2',
    #                'bert_bucket_layer-3',
    #                'bert_bucket_layer-4',
    #                'bert_bucket_layer-5',
    #                'bert_bucket_layer-6',
    #                'bert_bucket_layer-7',
    #                'bert_bucket_layer-8',
    #                'bert_bucket_layer-9',
    #                'bert_bucket_layer-10',
    #                'bert_bucket_layer-11',
    #                'bert_bucket_layer-12',
    #                'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer']}
#
    #
#
    #data = {model:[] for model in all_models}
    #value = 'maps_r2_'
    #nb_voxels = 26117
    #limit = limit_values[value]
    #x_labels = labels[1:]
    #
    #for key in plots.keys():
    #    models = plots[key]
    #    X = [surnames[model_name] for model_name in models]
    #    mean = np.zeros(len(models))
    #    error = np.zeros(len(models))
    #    mean_filtered = np.zeros(len(models))
    #    error_filtered = np.zeros(len(models))
    #    for index_mask in range(len(labels)-1):
    #        mask = math_img('img > 50', img=index_img(maps, index_mask))  
    #        masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
    #        masker.fit()
    #        for index_model, model in enumerate(models):
    #            y = [masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
    #            y_filtered = np.array([np.mean(array[array>np.percentile(array, 75)]) for array in y])
    #            y = np.array([np.mean(array) for array in y])
    #            error[index_model] = np.std(y)/np.sqrt(len(subjects))
    #            error_filtered[index_model] = np.std(y_filtered)/np.sqrt(len(subjects))
    #            mean_filtered[index_model] = np.mean(y_filtered)
    #            mean[index_model] = np.mean(y)
#
    #        # Plotting
    #        roi = labels[index_mask+1]
    #        save_folder = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', '3', key, surnames[value])
    #        layer_plot(X, mean, error, 
    #                    surnames, value, roi, limit, 
    #                    save_folder, 
    #                    key + '-' + roi + '-mean-'+ '-' + surnames[value] + '-' + 'averaged'  + '-all-voxels.png')
    #        
    #        layer_plot(X, mean_filtered, error_filtered,  
    #                    surnames, value, roi, limit, 
    #                    save_folder, 
    #                    key + '-' + roi + '-mean-'+ '-' + surnames[value] + '-' + 'averaged'  + '-filtered.png')
#
    #print("\t-->Done")


    #############################################################################
    ###################### Appendix  ####################
    #############################################################################
    #value_of_interest = 'maps_r2_'
    #print("Computing: Appendix...")
    #print("\tLooping through labeled masks...")
    #for index_mask in range(len(labels)-1):
    #    mask = math_img('img > 50', img=index_img(maps, index_mask))  
    #    masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
    #    masker.fit()
    #    to_compare = [['bert_bucket_all-layers', 'gpt2_all-layers'],
    #                    ['bert_bucket_pca_300_all-layers', 'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer'],
    #                    ['gpt2_pca_300_all-layers', 'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer'],
    #                    ['lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer', 'glove_embeddings']]
    #    for (model1, model2) in to_compare:
    #        tmp_path = os.path.join(paths['path2derivatives'], source, 'analysis', language, 'paper_plots', "Appendix")
    #        check_folder(os.path.join(tmp_path, 'averages'))
    #        distribution1 = masker.transform(os.path.join(paths['path2derivatives'], 'fMRI/ridge-indiv', language, 'averaged', model1, value_of_interest + '.nii.gz'))
    #        distribution2 = masker.transform(os.path.join(paths['path2derivatives'], 'fMRI/ridge-indiv', language, 'averaged', model2, value_of_interest + '.nii.gz'))
    #        save_hist(distribution1-distribution2, os.path.join(tmp_path, 'averages', 'hist_diff_averaged_{}_{}.png'.format(model1, model2)))
    #print("\t\t-->Done")
    #print("\t-->Done")
#
    #models = [['lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer', 'glove_embeddings'],
    #            ['bert_bucket_pca_300_all-layers', 'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer'],
    #            ['lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
    #            'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_cell_first-layer'],
    #            ['bert_bucket_all-layers',
    #            'gpt2_all-layers'], 
    #            ['bert_bucket_layer-8',
    #            'bert_bucket_layer-1'], 
    #            ['bert_bucket_layer-9',
    #            'bert_bucket_layer-8'],
    #            ['gpt2_layer-7',
    #            'gpt2_layer-1'], 
    #            ['gpt2_layer-8',
    #            'gpt2_layer-7']]
    
    #model1 = args.model1
    #model2 = args.model2
    #process_matrix(subjects, model1, model2, args.path2data1, args.path2data2, compute_significant=True)
