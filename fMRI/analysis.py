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

############################################################################
################################# Analysis #################################
############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nAnalysis of the fMRI pipeline.""")
    parser.add_argument("--subjects", nargs='+', action='append', default=[], help="Subjects list on whom we are running a test: list of 'sub-002...")
    parser.add_argument("--language", type=str, default='english', help="Language of the model.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Precise if we overwrite existing files.")
    parser.add_argument("--parallel", default=True, action='store_true', help="Precise if we run the code in parallel.")
    parser.add_argument("--roi_atlas", default=None, help="ROI atlas for parcellation study.")
    parser.add_argument("--path_yaml", default=None, help="Path to the yaml file with the models to compare.")

    args = parser.parse_args()
    subjects = args.subjects[0]
    source = 'fMRI'
    with open(args.path_yaml, 'r') as stream:
        try:
            analysis_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if not args.roi_atlas:
        atlas = datasets.fetch_atlas_harvard_oxford(params.atlas)
        labels = atlas['labels']
        maps = nilearn.image.load_img(atlas['maps'])

    # extract data
    i = 0
    for index_mask in range(len(labels)-1):
        mask = math_img('img > 50', img=index_img(maps, index_mask))  
        masker = NiftiMasker(mask_img=mask, memory='nilearn_cache', verbose=5)
        masker.fit()  # verify that it is well fitted and that we don't need to fit it with an image
        for analysis in analysis_list:
            for subject in subjects:
                subject = Subjects().get_subject(int(subject))
                model1 = [os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', args.language, analysis['model1'], 'ridge-indiv_{}_'.format(args.language) + analysis['model1'] + '_' + analysis['name'] + '_' + subject + '.nii.gz')]
                model2 = [os.path.join(paths.path2derivatives, 'fMRI/ridge-indiv', args.language, analysis['model2'], 'ridge-indiv_{}_'.format(args.language) + analysis['model2'] + '_' + analysis['name'] + '_' + subject + '.nii.gz')]
                analysis_title = analysis['title']
                x = masker.transform(model1) # you should include confounds
                y = masker.transform(model2)

                # save png comparison
                plt.figure(i)
                plt.scatter(x, y, c='red', marker='.')
                plt.scatter([np.mean(x)], [np.mean(y)], c='green', label='Average value')
                plt.scatter([np.percentile(x, 50)], [np.percentile(y, 50)], c='blue', label='Median value')
                plt.title('\n'.join(wrap('{} in {}'.format(analysis['name'], labels[index_mask+1]))))
                plt.xlabel('\n'.join(wrap('R2 of {}'.format(analysis['model1_name']))))
                plt.ylabel('\n'.join(wrap('R2 of {}'.format(analysis['model2_name']))))
                plt.xlim(0,0.2)
                plt.ylim(0,0.2)
                plt.plot([max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])], [max([np.min(x), np.min(y)]), min([np.max(x), np.max(y)])])
                plt.legend()
                plt.savefig(os.path.join(paths.path2derivatives, 'fMRI/analysis', analysis_title + ' - ' + labels[index_mask+1] + ' - ' + subject  + '.png'))
                i+=1