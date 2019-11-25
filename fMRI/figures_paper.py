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

def filter_distribution(distribution, percent):
    """ Keep the higgest percent% of the distribution,
    and padd the rest with nan values.
    """
    result = distribution[distribution>np.percentile(distribution, 75)]
    return distribution, result


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

def create_df_for_R(plots, plots_names, labels, subjects, maps, language, source, folder_name, surnames):
    for index, models in enumerate(plots):
        name = plots_names[index]
        x_labels = labels[1:]
        df_pearson_final = []
        df_r2_final = []
        df_significant_pearson_final = []
        df_significant_r2_final = []
        save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', folder_name, name)
        check_folder(save_folder)

        for subject in subjects:
            df_pearson = pd.DataFrame(data=[], columns=x_labels+['model_names'])
            df_r2 = pd.DataFrame(data=[], columns=x_labels+['model_names'])
            df_significant_pearson = pd.DataFrame(data=[], columns=x_labels+['model_names'])
            df_significant_r2 = pd.DataFrame(data=[], columns=x_labels+['model_names'])

            y_pearson = np.zeros((len(labels)-1, len(models)))
            y_r2 = np.zeros((len(labels)-1, len(models)))
            y_significant_pearson = np.zeros((len(labels)-1, len(models)))
            y_significant_r2 = np.zeros((len(labels)-1, len(models)))
            # Extract data
            for index_mask in range(len(labels)-1):
                mask = math_img('img > 50', img=index_img(maps, index_mask))  
                masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
                masker.fit()
                index_model = 0
                column = [[], [], [], []]
                model_names = []

                for model_name in models:
                    y_pearson[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model_name, subject, 'maps_pearson_corr')))
                    y_r2[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model_name, subject, 'maps_r2')))
                    y_significant_pearson[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model_name, subject, 'maps_significant_pearson_corr_with_pvalues')))
                    y_significant_r2[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model_name, subject, 'maps_significant_r2_with_pvalues')))
                    model_names.append(surnames[model_name])
                    column[0].append(y_pearson[index_mask, index_model])
                    column[1].append(y_r2[index_mask, index_model])
                    column[2].append(y_significant_pearson[index_mask, index_model])
                    column[3].append(y_significant_r2[index_mask, index_model])
                    index_model += 1
                df_pearson[x_labels[index_mask]] = column[0] ; df_pearson['model_names'] = model_names
                df_r2[x_labels[index_mask]] = column[1] ; df_r2['model_names'] = model_names
                df_significant_pearson[x_labels[index_mask]] = column[2] ; df_significant_pearson['model_names'] = model_names
                df_significant_r2[x_labels[index_mask]] = column[3] ; df_significant_r2['model_names'] = model_names
                df_pearson['subject'] = subject
                df_r2['subject'] = subject
                df_significant_pearson['subject'] = subject
                df_significant_r2['subject'] = subject
            df_pearson_final.append(df_pearson)
            df_r2_final.append(df_r2)
            df_significant_pearson_final.append(df_significant_pearson)
            df_significant_r2_final.append(df_significant_r2)
        df_pearson_final = pd.concat(df_pearson_final, axis=0)
        df_r2_final = pd.concat(df_r2_final, axis=0)
        df_significant_pearson_final = pd.concat(df_significant_pearson_final, axis=0)
        df_significant_r2_final = pd.concat(df_significant_r2_final, axis=0)
    
        df_pearson_final.drop_duplicates(inplace=True)
        df_pearson_final.to_csv(os.path.join(save_folder, 'pearson_data.csv'), index=False)
        df_r2_final.drop_duplicates(inplace=True)
        df_r2_final.to_csv(os.path.join(save_folder, 'r2_data.csv'), index=False)
        df_significant_pearson_final.drop_duplicates(inplace=True)
        df_significant_pearson_final.to_csv(os.path.join(save_folder, 'significant_pearson_data.csv'), index=False)
        df_significant_r2_final.drop_duplicates(inplace=True)
        df_significant_r2_final.to_csv(os.path.join(save_folder, 'significant_r2_data.csv'), index=False)

limit_values = {'maps_r2':0.1,
                'maps_pearson_corr':0.4,
                'maps_significant_pearson_corr_with_pvalues':0.4,
                'maps_significant_r2_with_pvalues':0.1}

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
                'maps_significant_r2_with_pvalues':'Significant R2'}


if __name__ == '__main__':

    paths2check = []
    source = 'fMRI'
    language = 'english'
    subjects = ['sub-057', 'sub-063', 'sub-067', 'sub-073', 'sub-077', 'sub-082', 'sub-101', 'sub-109', 'sub-110', 'sub-113', 'sub-114']
    inputs_path = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/'
    paths2check.append(inputs_path)

    # Sanity check
    for path in paths2check:
        check_folder(path)

    # retrieve default atlas  (= set of ROI)
    atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
    labels = atlas['labels']
    maps = nilearn.image.load_img(atlas['maps'])

    # Compute global masker
    fmri_runs = {}
    for subject in subjects:
        fmri_path = os.path.join(inputs_path, "data/fMRI/{language}/{subject}/func/")
        check_folder(fmri_path)
        fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path.format(language=language, subject=subject), 'fMRI_*run*')))
    masker = compute_global_masker(list(fmri_runs.values()))


    ###########################################################################
    ############### First plot (Comparison 300 features models) ###############
    ###########################################################################
    plots = [['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_cell_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_all-layers',
                'bert_bucket_pca_300_all-layers',
                'gpt2_pca_300_all-layers']]
    plots_names = ["Comparison 300 features models"]
    create_df_for_R(plots, plots_names, labels, subjects, maps, language, source, 'First plot (Comparison 300 features models)', surnames)

    ###########################################################################
    #################### Second plot (Individual features) ####################
    ###########################################################################
    plots = [['wordrate_word_position'],
                   ['wordrate_model'],
                   ['wordrate_log_word_freq'],
                   ['wordrate_function-word'],
                   ['wordrate_content-word'],
                   ['wordrate_all_model'],
                   ['topdown_model'],
                   ['rms_model'],
                   ['other_sentence_onset'],
                   ['mfcc_model'],
                   ['bottomup_model']]
    for object_of_interest in ['maps_r2', 'maps_pearson_corr','maps_significant_pearson_corr_with_pvalues', 'maps_significant_r2_with_pvalues']:
        for models in plots:
            all_paths = []
            for subject in subjects:
                model_name = models[0]
                path_template = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/outputs/maps/*{object_of_interest}*.nii.gz")
                path = sorted(glob.glob(path_template))[0]
                all_paths.append(path)
                img = load_img(path) 
                display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=True, plot_abs=False)
                save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "Second plots (Layer analysis)", model_name, surnames[object_of_interest], subject)
                check_folder(save_folder)
                display.savefig(os.path.join(save_folder, model_name + f' - {surnames[object_of_interest]} - ' + subject  + '.png'))
                display.close()
            data = [masker.transform(path) for path in all_paths]
            data = np.vstack(data)
            data = np.mean(data, axis=0)
            img = masker.inverse_transform(data)
            display = plot_glass_brain(img, display_mode='lzry', colorbar=True, black_bg=True, plot_abs=False)
            save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "Second plots (Layer analysis)", model_name, surnames[object_of_interest])
            check_folder(save_folder)
            display.savefig(os.path.join(save_folder, model_name + f' - {surnames[object_of_interest]} - ' + 'averaged across subjects'  + '.png'))
            display.close()

    # add best glass brain (bert/gpt2)

    ###########################################################################
    ################## Third plot (Layer analysis BERT/GPT2) ##################
    ###########################################################################
    i = 0
    plots = {"GPT2":['gpt2_embeddings',
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
                    'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_hidden_first-layer',
                    'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer'],
            "BERT":['bert_bucket_embeddings',
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
                    'lstm_wikikristina_embedding-size_600_nhid_768_nlayers_1_dropout_02_cell_first-layer']}
    for object_of_interest in ['maps_r2', 'maps_pearson_corr','maps_significant_pearson_corr_with_pvalues', 'maps_significant_r2_with_pvalues']:
        limit = limit_values[object_of_interest]
        for index_mask in range(len(labels)-1):
            mask = math_img('img > 50', img=index_img(maps, index_mask))  
            masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
            masker.fit()
            for key in plots.keys():
                models = plots[key]
                X = []
                Y_full = []
                Y_filtered = []
                for subject in subjects:
                    y = list(zip(*[list(filter_distribution(masker.transform(fetch_ridge_maps(model_name, subject, object_of_interest)), 75)) for model_name in models]))
                    x = [surnames[model_name] for model_name in models]
                    X.append(x)
                    Y_full.append([np.mean(i[0]) for i in y[0]])
                    Y_filtered.append([np.mean(i) for i in y[1]])
                    y_full = [np.mean(value[0]) for value in y[0]]
                    y_filtered = [np.mean(value) for value in y[1]]

                    plot = plt.plot(x, y_full)
                    plt.title('\n'.join(wrap(surnames[object_of_interest] + f' per ROI - {labels[index_mask+1]} - all voxels')))
                    plt.xlabel('Models')
                    plt.ylabel(surnames[object_of_interest])
                    plt.ylim(0,limit)
                    plt.xticks(rotation=30, fontsize=6, horizontalalignment='right')
                    plt.tight_layout()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "Third plot (Layer analysis {})".format(key), subject, surnames[object_of_interest])
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, key + labels[index_mask+1] + f' - {surnames[object_of_interest]} - ' + subject  + '_all-voxels.png'))
                    plt.close()

                    plot = plt.plot(x, y_filtered)
                    plt.title('\n'.join(wrap(surnames[object_of_interest] + f' per ROI - {labels[index_mask+1]} - top 25% voxels')))
                    plt.xlabel('Models')
                    plt.ylabel(surnames[object_of_interest])
                    plt.ylim(0,limit)
                    plt.xticks(rotation=30, fontsize=6, horizontalalignment='right')
                    plt.tight_layout()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', "Third plot (Layer analysis {})".format(key), subject, surnames[object_of_interest])
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, key + labels[index_mask+1] + f' - {surnames[object_of_interest]} - ' + subject  + '_top-25%-voxels.png'))
                    plt.close()
                Y_full = np.vstack(Y_full)
                Y_filtered = np.vstack(Y_filtered)
                X = X[0]

                error = np.std(Y_full, axis=0)/np.sqrt(len(subjects))
                Y_full = np.mean(Y_full, axis=0)
                plot = plt.plot(X, Y_full)
                plt.title('\n'.join(wrap(surnames[object_of_interest] + f' per ROI - {labels[index_mask+1]} - all voxels')))
                plt.xlabel('Models')
                plt.ylabel(surnames[object_of_interest])
                plt.ylim(0,limit)
                plt.xticks(rotation=30, fontsize=6, horizontalalignment='right')
                plt.errorbar(X, Y_full, error, linestyle='None', marker='^')
                plt.tight_layout()
                save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', 'Third plot (Layer analysis {})'.format(key), surnames[object_of_interest])
                check_folder(save_folder)
                plt.savefig(os.path.join(save_folder, key + labels[index_mask+1] + f' - {surnames[object_of_interest]} - ' + 'averaged across subjects'  + '_all-voxels.png'))
                plt.close()

                error = np.std(Y_filtered, axis=0)/np.sqrt(len(subjects))
                Y_filtered = np.mean(Y_filtered, axis=0)
                plot = plt.plot(X, Y_filtered)
                plt.title('\n'.join(wrap(surnames[object_of_interest] + f' per ROI - {labels[index_mask+1]} - top 25% voxels')))
                plt.xlabel('Models')
                plt.ylabel(surnames[object_of_interest])
                plt.ylim(0,limit)
                plt.xticks(rotation=30, fontsize=6, horizontalalignment='right')
                plt.errorbar(X, Y_filtered, error, linestyle='None', marker='^')
                plt.tight_layout()
                save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'paper_plots', 'Third plot (Layer analysis {})'.format(key), surnames[object_of_interest])
                check_folder(save_folder)
                plt.savefig(os.path.join(save_folder, key + labels[index_mask+1] + f' - {surnames[object_of_interest]} - ' + 'averaged across subjects'  + '_top-25%-voxels.png'))
                plt.close()

        

    ###########################################################################
    #################### Fourth plot (Layer analysis LSTM) ####################
    ###########################################################################
    plots = [['glove_embeddings',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_cell_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_all-layers',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_all-layers'],
                ['lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_hidden_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_2_dropout_02_cell_second-layer'],
                ['lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_hidden_third-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_3_dropout_02_cell_third-layer'],
                ['lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_third-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_hidden_fourth-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_first-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_second-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_third-layer',
                'lstm_wikikristina_embedding-size_600_nhid_75_nlayers_4_dropout_02_cell_fourth-layer']]
    plots_names = ["LSTM comparison with 300 units", "LSTM #L2 - layer comparison", "LSTM #L3 - layer comparison", "LSTM #L4 - layer comparison"]
    create_df_for_R(plots, plots_names, labels, subjects, maps, language, source, 'Fourth plot (Layer analysis LSTM)', surnames)

