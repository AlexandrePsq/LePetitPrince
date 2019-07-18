import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output
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
from textwrap import wrap 

import warnings
warnings.simplefilter(action='ignore')

params = Params()
paths = Paths()


if __name__ == '__main__':

    ###########################################################################
    ####################### Loading Yaml and parameters #######################
    ###########################################################################

    parser = argparse.ArgumentParser(description="""Objective:\nAnalysis of the fMRI pipeline.""")
    parser.add_argument("--path_yaml", default=None, help="Path to the yaml file with the models to compare.")
    parser.add_argument("--analysis", nargs='+', default=[], action='append', help='Analysis to perform.')

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

    paths.path2derivatives = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince/derivatives' # to delete


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
                    model1 = [os.path.join(paths.path2derivatives, source, 'ridge-indiv', language, analysis['model1'], 'ridge-indiv_{}_'.format(language) + analysis['model1'] + '_' + analysis['name'] + '_' + subject + '.nii.gz')]
                    model2 = [os.path.join(paths.path2derivatives, source, 'ridge-indiv', language, analysis['model2'], 'ridge-indiv_{}_'.format(language) + analysis['model2'] + '_' + analysis['name'] + '_' + subject + '.nii.gz')]
                    analysis_name = analysis['name']
                    x = masker.transform(model1) # you should include confounds
                    y = masker.transform(model2)

                    # save plots
                    plt.figure(i)
                    plt.scatter(x, y, c='red', marker='.')
                    plt.scatter([np.mean(x)], [np.mean(y)], c='green', label='Average value')
                    plt.scatter([np.percentile(x, 50)], [np.percentile(y, 50)], c='blue', label='Median value')
                    plt.title('\n'.join(wrap('{} in {}'.format(analysis['title'], labels[index_mask+1]))))
                    plt.xlabel('\n'.join(wrap('R2 of {}'.format(analysis['model1_name']))))
                    plt.ylabel('\n'.join(wrap('R2 of {}'.format(analysis['model2_name']))))
                    plt.xlim(0,0.2)
                    plt.ylim(0,0.2)
                    plt.plot([max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])], [max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])])
                    plt.legend()
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', 'scatter_plots')
                    check_folder(save_folder)
                    plt.savefig(os.path.join(save_folder, analysis_name + ' - ' + labels[index_mask+1] + ' - ' + subject  + '.png'))
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
                    save_folder = os.path.join(paths.path2derivatives, source, 'analysis', 'check_model')
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
        masker = NiftiMasker(memory='nilearn_cache', verbose=5)
        masker.fit()
        analysis_name = analysis['name']
        
        for model in analysis_parameters['model_complexity']:
            if model['variable_name'] != 'r2_test':
                x = model['complexity_variable']
                y_list = list(zip(*model['value_v-o-i'])) # list of list of subject values [sub1_list, sub2_list, ...]
                y = np.mean(model['value_v-o-i'], axis=1)
                plt.figure(i)
                plt.plot(x, y)
                plt.title('\n'.join(wrap(model['title'])))
                plt.xlabel('\n'.join(wrap(model['variable_name'])))
                plt.ylabel('\n'.join(wrap(model['variable_of_interest'])))
                plt.legend()
                plt.savefig(os.path.join(paths.path2derivatives, source, 'analysis', analysis_name + ' - ' + model['variable_of_interest'] + ' = f(' + model['variable_name'] + ') - ' + subject  + '.png'))
                plt.close()
                i+=1
            else:
                x = model['complexity_variable']
                y_list = []
                for subject in subjects:
                    y_sub = []
                    for nhid in model['complexity_variable']:
                        subject = Subjects().get_subject(int(subject))
                        # extract data
                        model_name = '_'.join([model['model_category'].lower(), 
                                                'wikikristina', 
                                                'embedding-size', model['parameters']['ninp'], 
                                                'nhid', model['parameters']['nhid'],
                                                'nlayers',  model['parameters']['nlayers'],
                                                'dropout',  model['parameters']['dropout'],
                                                model['parameters']['other']])
                        path = os.path.join(paths.path2derivatives, source, 'ridge-indiv', language, model_name)
                        file_name = '_'.join(['ridge-indiv', 
                                                language, model['model_category'].lower(),
                                                'wikikristina', 
                                                'embedding-size', model['parameters']['ninp'], 
                                                'nhid', model['parameters']['nhid'],
                                                'nlayers',  model['parameters']['nlayers'],
                                                'dropout',  model['parameters']['dropout'],
                                                model['parameters']['other'],
                                                model['variable_of_interest',
                                                'no_pca',
                                                subject + '.nii.gz'],
                                                ])
                        path2file = os.path.join(path, file_name)
                        y_sub.append(masker.transform(path2file))
                    plt.figure(i)
                    plt.boxplot(y_sub, positions=x)
                    plt.title('\n'.join(wrap(model['title'] + ' - ' + subject)))
                    plt.xlabel('\n'.join(wrap(model['variable_name'])))
                    plt.ylabel('\n'.join(wrap(model['variable_of_interest'])))
                    plt.legend()
                    plt.savefig(os.path.join(paths.path2derivatives, source, 'analysis', analysis_name + ' - ' + model['variable_of_interest'] + ' = f(' + model['variable_name'] + ') - ' + subject  + '.png'))
                    plt.close()
                    i += 1
                    y_list.append(y_sub)  # you should include confounds
                
                # save plots
                plt.figure(i)
                plt.boxplot(np.ndarray.tolist(np.mean(np.array(y_list), axis=0)), positions=x)
                plt.title('\n'.join(wrap(model['title'] + ' - ' + subject)))
                plt.xlabel('\n'.join(wrap(model['variable_name'])))
                plt.ylabel('\n'.join(wrap(model['variable_of_interest'])))
                plt.legend()
                plt.savefig(os.path.join(paths.path2derivatives, source, 'analysis', analysis_name + ' - ' + model['variable_of_interest'] + ' = f(' + model['variable_name'] + ') - ' + 'all-subjects'  + '.png'))
                plt.close()
                i += 1

                