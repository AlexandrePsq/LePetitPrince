import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output
from models.english.LSTM.tokenizer import tokenize

from itertools import combinations, product
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker
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
    path = os.path.join(paths.path2derivatives, 'fMRI', 'ridge-indiv', 'english', subject, model, 'outputs', 'maps', '*{}*.nii.gz'.format(value))
    files = sorted(glob.glob(path))
    return files[0]



if __name__ == '__main__':

    ###########################################################################
    ####################### Loading Yaml and parameters #######################
    ###########################################################################

    parser = argparse.ArgumentParser(description="""Objective:\nAnalysis of the fMRI pipeline.""")
    parser.add_argument("--path_yaml", default=None, help="Path to the yaml file with the models to compare.")
    parser.add_argument("--analysis", nargs='+', default=[], action='append', help='Analysis to perform.')
    parser.add_argument("--default_mask", default=os.path.join(paths.path2data, 'fMRI', 'english', 'sub-057', 'func', 'fMRI_english_sub-057_run1.nii.nii'), help='fMRI data to construct a global mask.')

    args = parser.parse_args()

    with open(args.path_yaml, 'r') as stream:
        try:
            analysis_parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    subjects = analysis_parameters['subjects']
    source = analysis_parameters['source']
    language = analysis_parameters['language']

    i = 0

    #paths.path2derivatives = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince/derivatives' # to delete
    #paths.path2data = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince/data'


    ###########################################################################
    ############################## Scatter plots ##############################
    ###########################################################################
    
    if 'scatter_plots' in args.analysis[0]:
        # retrieve default atlas  (= set of ROI)
        atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
        labels = atlas['labels']
        maps = nilearn.image.load_img(atlas['maps'])

        # extract data
        for index_mask in range(len(labels)-1):
            mask = math_img('img > 50', img=index_img(maps, index_mask))  
            masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
            masker.fit()
            for analysis in analysis_parameters['scatter_plots']:
                for subject in subjects:
                    subject = Subjects().get_subject(int(subject))
                    model1 = [os.path.join(paths.path2derivatives, source, analysis['input_data_folder1'], language, analysis['model1_folder'], analysis['model1']+ '_' + subject + '.nii.gz')]
                    model2 = [os.path.join(paths.path2derivatives, source, analysis['input_data_folder2'], language, analysis['model2_folder'], analysis['model2'] + '_' + subject + '.nii.gz')]
                    analysis_name = analysis['name']
                    x = masker.transform(model1) # you should include confounds
                    y = masker.transform(model2)

                    # save plots
                    plt.figure(i)
                    plt.scatter(x, y, c='red', marker='.')
                    plt.scatter([np.mean(x)], [np.mean(y)], c='green', label='Average value')
                    plt.scatter(x, y-x, c='black', label='increase in r2')
                    plt.scatter([np.percentile(x, 50)], [np.percentile(y, 50)], c='blue', label='Median value')
                    plt.title('\n'.join(wrap('{} in {}'.format(analysis['title'], labels[index_mask+1]))))
                    plt.xlabel('\n'.join(wrap('{}'.format(analysis['x_label']))))
                    plt.ylabel('\n'.join(wrap('{}'.format(analysis['y_label']))))
                    #plt.xlim(0,0.2)
                    #plt.ylim(0,0.2)
                    plt.plot([max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])], [max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])], c='blue')
                    plt.axhline(y=0., color='blue', linestyle='-')
                    plt.legend()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'scatter_plots', analysis_name)
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis['title'] + ' - ' + labels[index_mask+1] + ' - ' + subject  + '.png'))
                    plt.close()
                    i+=1
    

    ##########################################################################
    ############################## Check models ##############################
    ##########################################################################
    
    if 'check_model' in args.analysis[0]:
        # retrieve default atlas  (= set of ROI)
        atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
        labels = atlas['labels']
        maps = nilearn.image.load_img(atlas['maps'])

        # extract data
        for index_mask in range(len(labels)-1):
            mask = math_img('img > 50', img=index_img(maps, index_mask))  
            masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
            masker.fit()
            for analysis in analysis_parameters['check_model']:
                for subject in subjects:
                    subject = Subjects().get_subject(int(subject))
                    model = [os.path.join(paths.path2derivatives, source, 'ridge-indiv', language, analysis['model'], 'ridge-indiv_{}_'.format(language) + analysis['model'] + '_' + analysis['name'] + '_' + subject + '.nii.gz')]
                    analysis_name = analysis['name']
                    x = masker.transform(model).reshape(-1) # you should include confounds

                    # save plots
                    plt.figure(i)
                    plt.hist(x, color = 'blue', edgecolor = 'black', bins = 100)

                    plt.title('\n'.join(wrap('{} in {}'.format(analysis['title'], labels[index_mask+1]))))
                    plt.xlabel('\n'.join(wrap('R2 of {}'.format(analysis['model_name']))))
                    plt.ylabel('\n'.join(wrap('Density')))
                    plt.xlim(0,0.2)
                    plt.legend()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'check_model')
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + labels[index_mask+1] + ' - ' + subject  + '.png'))
                    plt.close()
                    i+=1
    




    ###########################################################################
    ################## Model complexity impact on regression ##################
    ###########################################################################
    # x: complexity variable
    # y_list: list of list of values for each subject [sub1_list, sub2_list, ...]
    #       sub1_list: list of values (perplexity, r2 distribution, ...) for a given subject
    if 'model_complexity' in args.analysis[0]:
        mask = mean_img(load_img(args.default_mask))
        mask = math_img('img > 50', img=mask)
        masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
        masker.fit()
        
        for analysis in analysis_parameters['model_complexity']:
            analysis_name = analysis['name']
            x = analysis['complexity_variable']
            if analysis['variable_of_interest'] not in ['r2_test', 'significative_r2_100']:
                y_list = list(zip(*analysis['value_v-o-i'])) # list of list of subject values [sub1_list, sub2_list, ...]
                y = np.mean(analysis['value_v-o-i'], axis=1)
                plt.figure(i)
                plt.plot(x, y)
                plt.title('\n'.join(wrap(analysis['title'])))
                plt.xlabel('\n'.join(wrap(analysis['variable_name'])))
                plt.ylabel('\n'.join(wrap(analysis['variable_of_interest'])))
                plt.legend()
                save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_complexity')
                check_folder(save_folder)
                plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + analysis['variable_of_interest'] + ' = f(' + analysis['variable_name'] + ') - ' + subject  + '.png'))
                plt.close()
                i+=1
            else:
                y_list = []
                for subject in subjects:
                    significant_values = [np.load(mask).sum() for mask in analysis['mask']]
                    max_values = []
                    y_sub = []
                    subject = Subjects().get_subject(int(subject))
                    for var in analysis['complexity_variable']:
                        # extract data
                        model_name = '_'.join([analysis['model_category'].lower(), 
                                                'wikikristina', 
                                                'embedding-size',  str(var if analysis['variable_name']=='ninp' else analysis['parameters']['ninp']),
                                                'nhid', str(var if analysis['variable_name']=='nhid' else analysis['parameters']['nhid']),
                                                'nlayers',   str(var if analysis['variable_name']=='nlayers' else analysis['parameters']['nlayers']),
                                                'dropout',  str(var if analysis['variable_name']=='dropout' else analysis['parameters']['dropout']).replace('.', ''),
                                                analysis['parameters']['which']])
                        path = os.path.join(paths.path2derivatives, source, 'ridge-indiv', language, model_name)
                        file_name = '_'.join(['ridge-indiv', 
                                                language, analysis['model_category'].lower(),
                                                'wikikristina', 
                                                'embedding-size', str(var if analysis['variable_name']=='ninp' else analysis['parameters']['ninp']), 
                                                'nhid', str(var if analysis['variable_name']=='nhid' else analysis['parameters']['nhid']),
                                                'nlayers',  str(var if analysis['variable_name']=='nlayers' else analysis['parameters']['nlayers']),
                                                'dropout',  str(var if analysis['variable_name']=='dropout' else analysis['parameters']['dropout']).replace('.', ''),
                                                analysis['parameters']['which'],
                                                analysis['variable_of_interest'],
                                                str('pca_' + str(var) if analysis['variable_name']=='pca' else analysis['parameters']['pca']),
                                                analysis['parameters']['voxel_wise'],
                                                subject + '.nii.gz'])
                        path2file = os.path.join(path, file_name)
                        y_sub.append(masker.transform(path2file)[0])
                        max_values.append(np.max(y_sub[-1]))
                    plt.figure(i)
                    plt.boxplot(y_sub, positions=x, sym='', widths=5, meanline=True, showmeans=True)
                    plt.title('\n'.join(wrap(analysis['title'] + ' - ' + subject)))
                    plt.xlabel('\n'.join(wrap(analysis['variable_name'])))
                    plt.ylabel('\n'.join(wrap(analysis['variable_of_interest'])))
                    plt.legend()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_complexity')
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + analysis['variable_of_interest'] + ' = f(' + analysis['variable_name'] + ') - ' + subject  + '.png'))
                    plt.close()
                    i += 1

                    fig, ax1 = plt.subplots()
                    plt.title('\n'.join(wrap('Count R2>0 + R2 max' ' - ' + subject)))

                    color = 'tab:red'
                    ax1.set_xlabel('\n'.join(wrap(analysis['variable_name'])))
                    ax1.set_ylabel('\n'.join(wrap('count')), color=color)
                    ax1.plot(x, significant_values, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                    color = 'tab:blue'
                    ax2.set_ylabel('\n'.join(wrap('R2 max')), color=color)  # we already handled the x-label with ax1
                    ax2.plot(x, max_values, color=color)
                    ax2.tick_params(axis='y', labelcolor=color)

                    fig.tight_layout()
                    plt.legend()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_complexity')
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + analysis['variable_of_interest'] + ' non zero values count' + ' - ' + subject  + '.png'))
                    plt.close()

                    i += 1
                    y_list.append(y_sub)  # you should include confounds



                # save plots
                plt.figure(i)
                plt.boxplot(np.ndarray.tolist(np.mean(np.array(y_list), axis=0)), positions=x)
                plt.title('\n'.join(wrap(analysis['title'] + ' - ' + subject)))
                plt.xlabel('\n'.join(wrap(analysis['variable_name'])))
                plt.ylabel('\n'.join(wrap(analysis['variable_of_interest'])))
                plt.legend()
                save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_complexity')
                check_folder(save_folder)
                plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + analysis['variable_of_interest'] + ' = f(' + analysis['variable_name'] + ') - ' + 'averaged_accross_subjects'  + '.png'))
                plt.close()
                plt.close()
                i += 1

    ###########################################################################
    ############################ Specific analysis ############################
    ###########################################################################
    if 'specific_analysis' in args.analysis[0]:
        window_size_beg = 0
        window_size_end = 20
        color_map = {'entropy': 'tab:red',
                        'surprisal': 'tab:blue'}
        analysis_name = 'Entropy - Surprisal'
        for subject in subjects:
            subject = Subjects().get_subject(int(subject))
            for model in analysis_parameters['specific_analysis']:
                for run in range(1, params.nb_runs + 1):
                    path = os.path.join(paths.path2data, model['data'].format(run))
                    iterator = tokenize(path, language)
                    x = np.arange(len(iterator))[window_size_beg:window_size_end]

                    def get_path(name, model):
                        model_name = '_'.join([model['parameters']['model_category'].lower(), 
                                                    'wikikristina', 
                                                    'embedding-size',  str(model['parameters']['ninp']),
                                                    'nhid', str(model['parameters']['nhid']),
                                                    'nlayers',   str(model['parameters']['nlayers']),
                                                    'dropout',  str(model['parameters']['dropout']).replace('.', ''),
                                                    model['parameters']['other'].format(name)])
                        path = os.path.join(paths.path2derivatives, source, 'raw-features', language, model_name)
                        file_name = '_'.join(['raw-features', 
                                                language, model['parameters']['model_category'].lower(),
                                                'wikikristina', 
                                                'embedding-size',  str(model['parameters']['ninp']),
                                                'nhid', str(model['parameters']['nhid']),
                                                'nlayers',   str(model['parameters']['nlayers']),
                                                'dropout',  str(model['parameters']['dropout']).replace('.', ''),
                                                model['parameters']['other'].format(name),
                                                'run{}.csv'.format(run)])
                        return os.path.join(path, file_name)

                    y_ent = pd.read_csv(get_path('entropy', model))['entropy'][window_size_beg:window_size_end]
                    y_sur = pd.read_csv(get_path('surprisal', model))['surprisal'][window_size_beg:window_size_end]
                    
                    fig, ax1 = plt.subplots()
                    plt.xticks(x, iterator[window_size_beg:window_size_end], rotation=90)
                    plt.title('\n'.join(wrap('Entropy & Surprisal' + ' - ' + subject)))
                    color = color_map['entropy']
                    ax1.set_xlabel('\n'.join(wrap('Le Petit Prince text')))
                    ax1.set_ylabel('\n'.join(wrap('Entropy')), color=color)
                    ax1.plot(x, y_ent, color=color)
                    plt.legend()
                    ax1.tick_params(axis='y', labelcolor=color)

                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                    color = color_map['surprisal']
                    ax2.set_ylabel('\n'.join(wrap('Surprisal')), color=color)  # we already handled the x-label with ax1
                    ax2.plot(x, -y_sur, color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
                    fig.tight_layout()
                    plt.legend()
                    ax = plt.gca()
                    ax.grid(True, which='both')
                    # plt.grid(which='both')
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'specific_analysis')
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + 'window_size_' + str(window_size_end-window_size_beg) + ' - ' + subject  + 'run{}.png'.format(run)))
                    plt.close()
                    i += 1

    ##########################################################################
    ############################ Model Comparison ############################
    ##########################################################################
    
    if 'model_comparison' in args.analysis[0]:
        # retrieve default atlas  (= set of ROI)
        atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
        labels = atlas['labels']
        maps = nilearn.image.load_img(atlas['maps'])
        analysis_name = 'Model comparison - Average per ROI - Pearson & R2'

        x = labels[:-1]
        for analysis in analysis_parameters['model_comparison']:
            y_pearson = np.zeros((len(labels)-1, len(analysis['models'])))
            y_r2 = np.zeros((len(labels)-1, len(analysis['models'])))
            # extract data
            for index_mask in range(len(labels)-1):
                mask = math_img('img > 50', img=index_img(maps, index_mask))  
                masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
                masker.fit()
                index_model = 0
                for model in analysis['models']:
                    for subject in subjects:
                        subject = Subjects().get_subject(int(subject))
                        y_pearson[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model, subject, 'maps_pearson_corr')))
                        y_r2[index_mask, index_model] = np.mean(masker.transform(fetch_ridge_maps(model, subject, 'maps_r2')))
                        index_model += 1

            # save plots
            plt.figure(2*i)
            plot = plt.plot(x, y_pearson)
            plt.title('Pearson coefficient per ROI')
            plt.xlabel('Regions of interest (ROI)')
            plt.ylabel('Pearson coefficient value')
            plt.legend(plot, [model for model in analysis['models']], loc=1)
            save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_comparison', analysis_name)
            check_folder(save_folder)
            plt.savefig(os.path.join(save_folder, analysis['title'] + ' - pearson - ' + subject  + '.png'))
            plt.close()

            plt.figure(2*i + 1)
            plot = plt.plot(x, y_r2)
            plt.title('R2 per ROI')
            plt.xlabel('Regions of interest (ROI)')
            plt.ylabel('R2 value')
            plt.legend(plot, [model for model in analysis['models']], loc=1)
            save_folder = os.path.join(paths.path2derivatives, source, 'analysis', language, 'model_comparison', analysis_name)
            check_folder(save_folder)
            plt.savefig(os.path.join(save_folder, analysis['title'] + ' - R2 - ' + subject  + '.png'))
            plt.close()
            i+=1
