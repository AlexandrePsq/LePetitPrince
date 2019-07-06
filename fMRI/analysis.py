import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Rois, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output
from itertools import combinations, product
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
from nilearn.input_data import NiftiMapsMasker
from nilearn.regions import RegionExtractor
from nilearn.plotting import plot_glass_brain
from utilities.utils import get_data, get_output_parent_folder, check_folder, transform_design_matrices, pca
from utilities.first_level_analysis import compute_global_masker
from nilearn.masking import compute_epi_mask

from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import yaml
import nibabel as nib
import sklearn
import pandas as pd
from tqdm import tqdm
from .utils import get_r2_score, log
from .settings import Params, Paths
import pickle
from os.path import join
import argparse

import warnings
warnings.simplefilter(action='ignore')


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
        model2compare = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

    extraction = RegionExtractor(args.roi_atlas, min_region_size=3200,
                             threshold=98, thresholding_strategy='percentile', extractor='connected_components')
    

    # extract data 
    masker = NiftiMapsMasker(maps_img=args.roi_atlas, standardize=False,
                            memory='nilearn_cache', verbose=5)
    masker.fit()  # verify that it is well fitted and that we don't need to fit it with an image
    for action in model2compare.keys():
        x = masker.transform(action[0])   # confounds=data.confounds ??
        y = masker.transform(action[1])   # path to the ridge map nii

    # save it into a pandas DataFrame
    df = pd.DataFrame(columns=['subject', 'con', 'ROI', 'beta'])

    n1, n2 = x.shape
    k = 0
    for i1 in range(n1):
        for i2 in range(n2):
            plt.scatter(x[i1, i2], y1[i1, i2], c = 'red')
            plt.title('scatter plot')

            plt.xlabel('eigenvalue number')
            plt.ylabel('explained variance (%)')
            plt.axhline(y=var_model, color='g', linestyle='--', label='variance explained by the model: {0:.2f}%'.format(var_model))
            plt.axvline(x=n_components, color='r', linestyle='--', label='number of components: {}'.format(n_components))
            plt.title('\n'.join(wrap(data_name)))
            plt.legend()
            plt.savefig(os.path.join(paths.path2derivatives, 'fMRI', data_name+ '_pca.png'))
             df.loc[k] = pd.Series({'subject': subj[i1],
                                    'con': con[i1],
                                    'ROI': roi_names[i2],
                                    'beta': values[i1, i2]})
             k = k + 1
    df.to_csv(output, index=False)

#####################################################################
############ Model comparison with equivalent perplexity ############
#####################################################################
params = Params()


plotting.plot_prob_atlas(atlas_filename, cut_coords=[-55, -20, 8])
for index in range(39):
    plot_glass_brain(nilearn.image.index_img(extraction.regions_img_,index))
plotting.show()

# min_region_size in voxel volume mm^3


# Just call fit() to execute region extraction procedure
extraction.fit()
regions_img = extraction.regions_img_

######################################################################
############# Model comparison low-level VS high-level ###############
######################################################################





#####################################################################
######### Model comparison with different number of layers ##########
#####################################################################



x = [1, 2, 3, 4, 5]
y1 = [1, 2, 3, 4, 5]
y2 = [1, 4, 9, 16, 25]
plt.scatter(x, y1, c = 'red')
plt.scatter(x, y2, c = 'yellow')
plt.title('scatter plot')

plt.xlabel('eigenvalue number')
plt.ylabel('explained variance (%)')
plt.axhline(y=var_model, color='g', linestyle='--', label='variance explained by the model: {0:.2f}%'.format(var_model))
plt.axvline(x=n_components, color='r', linestyle='--', label='number of components: {}'.format(n_components))
plt.title('\n'.join(wrap(data_name)))
plt.legend()
plt.savefig(os.path.join(paths.path2derivatives, 'fMRI', data_name+ '_pca.png'))