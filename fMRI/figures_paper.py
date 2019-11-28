import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output
from models.LSTM.tokenizer import tokenize

from itertools import combinations, product
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker, MultiNiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_glass_brain, plot_img
import nibabel as nib
from nilearn.regions import RegionExtractor
from utilities.utils import get_data, get_output_parent_folder, check_folder, transform_design_matrices, pca

from joblib import Parallel, delayed
import yaml
import pandas as pd
import argparse
import glob
from textwrap import wrap 

import warnings
warnings.simplefilter(action='ignore')

params = Params()
paths = Paths()


def fetch_ridge_maps(model, subject, value):
    """Retrieve the R2/pearson values (significant or not)
    for a given subject and model.
    """
    path = os.path.join(paths.path2derivatives, 'fMRI', 'ridge-indiv', 'english', subject, model, 'outputs', 'maps', '*{}*.nii.gz'.format(value))
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

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    """Return a MultiNiftiMasker object.
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, smoothing_fwhm=5) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def save_hist(distribution, path):
    plt.hist(distribution[~np.isnan(distribution)], bins=50)
    plt.savefig(path)
    plt.close()

def vertical_plot(data, x_names, analysis_name, save_folder, object_of_interest, surnames, models, syntactic_roi, language_roi, figsize=(10,12)):
    """Plots vertically the 6 first models.
    """
    limit = (-0.02, 0.06) if object_of_interest=='maps_r2' else None
    dash_inf = limit[0] if limit else 0
    x = x_names.copy()
    plt.figure(figsize=figsize)
    ax = plt.axes()
    order = np.argsort(np.mean(data, axis=1))
    data = data[order, :]
    x = [x[i] for i in order]
    ax = plt.axes()
    plot = plt.plot(data[:,0], x, '.b',
                    data[:,1], x, '.y',
                    data[:,2], x, '.r',
                    data[:,3], x, '.m',
                    data[:,4], x, '.k',
                    data[:,5], x, '.g')
    plt.title('R2 per ROI')
    plt.ylabel('Regions of interest (ROI)')
    plt.xlabel('R2 value')
    plt.hlines(x, dash_inf, data[:,0], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,1], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,2], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,3], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,4], linestyle="dashed", alpha=0.1)
    plt.hlines(x, dash_inf, data[:,5], linestyle="dashed", alpha=0.1)
    plt.xlim(limit)
    plt.legend(plot, [surnames[model] for model in models], ncol=3, bbox_to_anchor=(0,0,1,1), fontsize=5)
    for index, label in enumerate(x):
        if label in language_roi:
            if label in syntactic_roi:
                ax.get_yticklabels()[index].set_bbox(dict(facecolor="red", alpha=0.6))
            ax.axhspan(index-0.5, index+0.5, alpha=0.1, color='red')
    plt.tight_layout()
    check_folder(save_folder)
    plt.savefig(os.path.join(save_folder, f'{analysis_name}-{object_of_interest}.png'))
    plt.close()

def layer_plot(X,Y, error, surnames, object_of_interest, roi, limit, save_folder, name):
    """Layer-wised plot per ROI
    """
    plot = plt.plot(X, Y)
    plt.title('\n'.join(wrap(surnames[object_of_interest] + f' per ROI - {roi} - all voxels')))
    plt.xlabel('Models')
    plt.ylabel(surnames[object_of_interest])
    plt.ylim(0,limit)
    plt.xticks(rotation=30, fontsize=6, horizontalalignment='right')
    plt.errorbar(X, Y, error, linestyle='None', marker='^')
    plt.tight_layout()
    check_folder(save_folder)
    plt.savefig(os.path.join(save_folder, name))
    plt.close()

all_models = ['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
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

limit_values = {'maps_r2':0.08,
                'maps_pearson_corr':0.4,
                'maps_significant_pearson_corr_with_pvalues':0.4,
                'maps_significant_r2_with_pvalues':0.08}

syntactic_roi = ['Frontal Pole',
'Superior Frontal Gyrus',
'Inferior Frontal Gyrus, pars triangularis',
'Inferior Frontal Gyrus, pars opercularis',
'Superior Temporal Gyrus, anterior division',
'Superior Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, anterior division',
'Middle Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, temporooccipital part',
'Inferior Temporal Gyrus, anterior division',
'Inferior Temporal Gyrus, posterior division',
'Inferior Temporal Gyrus, temporooccipital part',
'Supramarginal Gyrus, anterior division',
'Supramarginal Gyrus, posterior division',
'Angular Gyrus',
'Frontal Medial Cortex']

language_roi = ['Frontal Pole',
'Insular Cortex',
'Superior Frontal Gyrus',
'Middle Frontal Gyrus',
'Inferior Frontal Gyrus, pars triangularis',
'Inferior Frontal Gyrus, pars opercularis',
'Temporal Pole',
'Superior Temporal Gyrus, anterior division',
'Superior Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, anterior division',
'Middle Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, temporooccipital part',
'Inferior Temporal Gyrus, anterior division',
'Inferior Temporal Gyrus, posterior division',
'Inferior Temporal Gyrus, temporooccipital part',
'Supramarginal Gyrus, anterior division',
'Supramarginal Gyrus, posterior division',
'Angular Gyrus',
'Frontal Medial Cortex',
'Paracingulate Gyrus',
'Cingulate Gyrus, anterior division',
'Cingulate Gyrus, posterior division']

surnames = {'wordrate_word_position': 'Word position',
                'wordrate_model': 'Wordrate',
                'wordrate_log_word_freq': 'Log word frequency',
                'wordrate_function-word': 'Function words',
                'wordrate_content-word': 'Content words',
                'wordrate_all_model': 'All word-related models',
                'topdown_model': 'Topdown parser',
                'rms_model': 'RMS',
                'other_sentence_onset': 'Sentence onset',
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
                'bottomup_model': 'Bottomup parser',
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
                'bert_bucket_pca_300_all-layers': 'BERT-all-pca-300',
                'gpt2_pca_300_all-layers':'GPT2-all-pca-300',
                'maps_r2':'R2', 
                'maps_pearson_corr':'Pearson coefficient',
                'maps_significant_pearson_corr_with_pvalues':'Significant Pearson', 
                'maps_significant_r2_with_pvalues':'Significant R2',
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
                'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'SMC', 
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

    paths2check = []
    source = 'fMRI'
    language = 'english'
    subjects = ['sub-057', 'sub-063', 'sub-067', 'sub-073', 'sub-077', 'sub-082', 'sub-101', 'sub-109', 'sub-110', 'sub-113', 'sub-114']
    inputs_path = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince/' #'/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    paths2check.append(inputs_path)

    # Sanity check
    for path in paths2check:
        check_folder(path)

    # retrieve default atlas  (= set of ROI)
    atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
    labels = atlas['labels']
    labels = [surnames[value] for value in labels]
    maps = nilearn.image.load_img(atlas['maps'])

    # Compute global masker
    fmri_runs = {}
    print("Commuting global masker...")
    for subject in subjects:
        fmri_path = os.path.join(inputs_path, "data/fMRI/{language}/{subject}/func/")
        check_folder(fmri_path)
        fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path.format(language=language, subject=subject), 'fMRI_*run*')))
    global_masker = compute_global_masker(list(fmri_runs.values()))
    print("\t-->Done")


    ###########################################################################
    #################### 1 (Individual features) ####################
    ###########################################################################
    #print("Computing: 1 (Individual features)")
    #plots = [['wordrate_word_position'],
    #               ['wordrate_model'],
    #               ['wordrate_log_word_freq'],
    #               ['wordrate_function-word'],
    #               ['wordrate_content-word'],
    #               ['wordrate_all_model'],
    #               ['topdown_model'],
    #               ['rms_model'],
    #               ['other_sentence_onset'],
    #               ['mfcc_model'],
    #               ['bottomup_model'],
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
    #               ['lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer'],
    #               ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer'],
    #                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer'],
    #                ['glove_embeddings']]
    #for object_of_interest in ['maps_r2', 'maps_significant_r2_with_pvalues']:
    #    for models in plots:
    #        all_paths = []
    #        for subject in subjects:
    #            model_name = models[0]
    #            path_template = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/outputs/maps/*{object_of_interest}*.nii.gz")
    #            path = sorted(glob.glob(path_template))[0]
    #            all_paths.append(path)
    #            img = load_img(path) 
    #            display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=False, plot_abs=False)
    #            save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "1", model_name, surnames[object_of_interest], subject)
    #            check_folder(save_folder)
    #            display.savefig(os.path.join(save_folder, model_name + f' - {surnames[object_of_interest]} - ' + subject  + '.png'))
    #            display.close()
    #        data = [global_masker.transform(path) for path in all_paths]
    #        data = np.vstack(data)
    #        data = np.mean(data, axis=0)
    #        img = global_masker.inverse_transform(data)
    #        display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=False, plot_abs=False)
    #        save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "1", model_name, surnames[object_of_interest])
    #        check_folder(save_folder)
    #        display.savefig(os.path.join(save_folder, model_name + f' - {surnames[object_of_interest]} - ' + 'averaged across subjects'  + '.png'))
    #        display.close()
    #print("\t\t-->Done")

    ###########################################################################
    ############### 2 (Comparison 300 features models) ###############
    ###########################################################################
    print("Computing: 2 (Comparison 300 features models)...")
    models = ['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
                'bert_bucket_pca_300_all-layers',
                'gpt2_pca_300_all-layers',
                'bert_bucket_all-layers',
                'gpt2_all-layers']
    plot_name = ["Comparison 300 features models"]
    data = {model:[] for model in all_models}
    nb_voxels = 26117
    table = np.zeros((len(all_models), nb_voxels))
    for value in ['maps_r2']:
        tmp_path = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "2")
        check_folder(os.path.join(tmp_path, 'averages'))

        print("\tTransforming data...")
        for model_index, model in enumerate(all_models):
            data[model] = [global_masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
            distribution = np.mean(np.vstack(data[model]), axis=0)
            table[model_index, distribution>np.percentile(distribution, 75)] = 1
            save_hist(distribution, os.path.join(tmp_path, 'averages', f'hist_averaged_{value}_{model}.png'))
            path2output_raw = os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model)
            check_folder(path2output_raw)
            nib.save(global_masker.inverse_transform(distribution), os.path.join(path2output_raw, value + '.nii.gz'))
        print("\t\t-->Done")

        table = np.sum(table, axis=0)
        mask = (table>np.percentile(table, 75))
        print("\tSaving filtered distributions per model and subject...")
        for model in all_models:
            for index, subject in enumerate(subjects):
                distribution = data[model][index][0]
                distribution[~mask] = np.nan
                path2output_raw = os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, subject, model, 'outputs/maps')
                check_folder(path2output_raw)
                nib.save(global_masker.inverse_transform(distribution), os.path.join(path2output_raw, model + '_outputs_maps_' + 'filtered_25%_' + value + 'pca_None_voxel_wise_'+ subject + '.nii.gz'))
            distribution = np.mean(np.vstack(data[model]), axis=0)
            distribution[~mask] = np.nan
            path2output_raw = os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model)
            check_folder(path2output_raw)
            nib.save(global_masker.inverse_transform(distribution), os.path.join(path2output_raw, 'filtered_25%_' + value + '.nii.gz'))
        print("\t\t-->Done")

        x_labels = labels[1:]
        mean = np.zeros((len(labels)-1, len(models)))
        median = np.zeros((len(labels)-1, len(models)))
        percentile = np.zeros((len(labels)-1, len(models)))
        mean_filtered = np.zeros((len(labels)-1, len(models)))
        median_filtered = np.zeros((len(labels)-1, len(models)))
        percentile_filtered = np.zeros((len(labels)-1, len(models)))
        # extract data
        print("\tLooping through labeled masks...")
        for index_mask in range(len(labels)-1):
            mask = math_img('img > 50', img=index_img(maps, index_mask))  
            masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
            masker.fit()

            for index_model, model in enumerate(models):
                path2output_raw = os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model)
                array = masker.transform(os.path.join(path2output_raw, value + '.nii.gz'))
                mean[index_mask, index_model] = np.mean(array)
                median[index_mask, index_model] = np.percentile(array, 50)
                percentile[index_mask, index_model] = np.percentile(array, 75)

                path2output_raw = os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model)
                array = masker.transform(os.path.join(path2output_raw, 'filtered_25%_' + value + '.nii.gz'))
                array = array[~np.isnan(array)]
                mean_filtered[index_mask, index_model] = np.mean(array)
                median_filtered[index_mask, index_model] = np.percentile(array, 50)
                percentile_filtered[index_mask, index_model] = np.percentile(array, 75)
        print("\t\t-->Done")

        print("\tPlotting...")
        # save plots
        vertical_plot(mean, x_labels, 'Mean-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        vertical_plot(median, x_labels, 'Median-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        vertical_plot(percentile, x_labels, 'Third-quartile-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        vertical_plot(mean_filtered, x_labels, 'Mean-filtered-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        vertical_plot(median_filtered, x_labels, 'Median-filtered-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        vertical_plot(percentile_filtered, x_labels, 'Third-quartile-filtered-comparison_300_features', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2'), 
                        value, surnames, models, syntactic_roi, language_roi)
        print("\t\t-->Done")

    ###########################################################################
    ############### 2 bis (avec significant R2 values) ###############
    ###########################################################################
    print("Dealing with significant R2...")
    value = 'maps_significant_r2_with_pvalues'
    counter = np.zeros((len(labels)-1, len(models)))
    mean = np.zeros((len(labels)-1, len(models)))
    for index_mask in range(len(labels)-1):
        mask = math_img('img > 50', img=index_img(maps, index_mask))  
        masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
        masker.fit()
        for index_model, model in enumerate(models):
            data = [masker.transform(fetch_ridge_maps(model, subject, value)) for subject in subjects]
            counter[index_mask, index_model] = np.mean([np.count_nonzero(~np.isnan(array)) for array in data])
            mean[index_mask, index_model] = np.mean([np.mean(array) for array in data])
    vertical_plot(counter, x_labels, 'Count of non-Nan values per ROI', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2bis'), 
                        value, surnames, models, syntactic_roi, language_roi)
    vertical_plot(mean, x_labels, 'Mean significant R2', 
                        os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '2bis'), 
                        value, surnames, models, syntactic_roi, language_roi)
    print("\t-->Done")


    ###########################################################################
    ############### 3 (Layer-wise analysis) ###############
    ###########################################################################
    #print("Computing: 3 (Layer-wise analysis)...")
    #object_of_interest = 'maps_r2'
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
    #for object_of_interest in ['maps_r2', 'maps_significant_r2_with_pvalues']:
    #    limit = limit_values[object_of_interest]
    #    for index_mask in range(len(labels)-1):
    #        mask = math_img('img > 50', img=index_img(maps, index_mask))  
    #        masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=0)
    #        masker.fit()
    #        for key in plots.keys():
    #            models = plots[key]
    #            Y_full = []
    #            Y_filtered_25 = [] if object_of_interest=='maps_r2' else None
    #            X = [surnames[model_name] for model_name in models]
    #            for subject in subjects:
    #                y = list(zip(*[list((masker.transform(fetch_ridge_maps(model_name, subject, object_of_interest)), masker.transform(fetch_ridge_maps(model_name, subject, 'filtered_25%_' + object_of_interest )))) for model_name in models]))
    #                Y_full.append([np.mean(i[0]) for i in y[0]])
    #                if object_of_interest=='maps_r2':
    #                    Y_filtered_25.append([np.mean(i[0]) for i in y[1]])
#
    #            Y_full = np.vstack(Y_full)
    #            Y_filtered_25 = np.vstack(Y_filtered_25) if object_of_interest=='maps_r2' else None
    #            
    #            # Plotting
    #            roi = labels[index_mask+1]
    #            save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', '3', key, surnames[object_of_interest])
    #            layer_plot(X, np.mean(Y_full, axis=0), np.std(Y_full, axis=0)/np.sqrt(len(subjects)), 
    #                        surnames, object_of_interest, roi, limit, 
    #                        save_folder, 
    #                        key + '_' + roi + '_mean_'+ f' - {surnames[object_of_interest]} - ' + 'averaged_across_subjects'  + '_all-voxels.png')
    #            
    #            if object_of_interest=='maps_r2':
    #                error = np.std(Y_filtered_25, axis=0)/np.sqrt(len(subjects))
    #                layer_plot(X, np.mean(Y_filtered_25, axis=0), error, 
    #                            surnames, object_of_interest, roi, limit, 
    #                            save_folder, 
    #                            key + '_' + roi + '_mean_'+ f' - {surnames[object_of_interest]} - ' + 'averaged_across_subjects'  + '_top-25%-voxels.png')
    #                layer_plot(X, np.median(Y_filtered_25, axis=0), error, 
    #                            surnames, object_of_interest, roi, limit, 
    #                            save_folder, 
    #                            key + '_' + roi + '_median_'+ f' - {surnames[object_of_interest]} - ' + 'averaged_across_subjects'  + '_top-25%-voxels.png')
    #                layer_plot(X, np.percentile(Y_filtered_25, 75, axis=0), error, 
    #                            surnames, object_of_interest, roi, limit, 
    #                            save_folder, 
    #                            key + '_' + roi + '_third-quartile_'+ f' - {surnames[object_of_interest]} - ' + 'averaged_across_subjects'  + '_top-25%-voxels.png')
#
    #print("\t-->Done")
#
#
    ############################################################################
    ##################### Appendix  ####################
    ############################################################################
    #value_of_interest = 'maps_r2'
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
    #        tmp_path = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "Appendix")
    #        check_folder(os.path.join(tmp_path, 'averages'))
    #        distribution1 = masker.transform(os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model1, value_of_interest + '.nii.gz'))
    #        distribution2 = masker.transform(os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', language, 'averaged', model2, value_of_interest + '.nii.gz'))
    #        save_hist(distribution1-distribution2, os.path.join(tmp_path, 'averages', f'hist_diff_averaged_{model1}_{model2}.png'))
    #print("\t\t-->Done")
    #print("\t-->Done")