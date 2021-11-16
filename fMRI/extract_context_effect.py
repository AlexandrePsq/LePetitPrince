import warnings
warnings.simplefilter(action='ignore')

import os
import gc
import glob
import itertools
import argparse
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed

import scipy
import nistats
import numpy as np
import pandas as pd
#import pingouin as pg
from scipy.stats import norm
import statsmodels.api as sm 
from sklearn import manifold
from scipy.special import erf
from scipy.optimize import curve_fit
from statsmodels.formula.api import ols
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.manifold import LocallyLinearEmbedding
from statsmodels.stats.multitest import multipletests
from nistats.second_level_model import SecondLevelModel
from sklearn.cluster import AgglomerativeClustering, KMeans


import nibabel as nib
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img, new_img_like
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker, MultiNiftiMasker
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from nilearn._utils.niimg import _safe_get_data
from nilearn import plotting
from nilearn import datasets
from scipy.stats import norm
from nilearn.surface import vol_to_surf

import utils 
import reporting
from logger import Logger
from splitter import Splitter
from linguistics_info import *

language = 'english'

PROJECT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
OUTPUT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/{language}"
INPUT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/stimuli-representations/{language}"
FMRIDATA_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI/{language}"
MASKER_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ROI_masks/global_masker_95%_{language}"
ALL_MASKS_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ROI_masks/"
SAVING_FOLDER = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/clustering/{language}"
TMP_FOLDER = f"/home/ap259944/tmp"


logger = Logger(os.path.join(PROJECT_PATH, 'logs.txt'))

global_masker_50 = reporting.fetch_masker(f"{ALL_MASKS_PATH}/global_masker_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)
global_masker_95 = reporting.fetch_masker(f"{ALL_MASKS_PATH}/global_masker_95%_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)


original_masker = global_masker_50
new_masker = global_masker_95
original_masker.set_params(detrend=False, standardize=False)
new_masker.set_params(detrend=False, standardize=False)

params = global_masker_95.get_params()
new_img = new_img_like(global_masker_95.mask_img, scipy.ndimage.binary_dilation(global_masker_95.mask_img.get_data()))
dilated_global_masker_95 = NiftiMasker()
dilated_global_masker_95.set_params(**params)
dilated_global_masker_95.mask_img = new_img
dilated_global_masker_95.fit()

atlas_maps, labels = reporting.load_atlas()
subject_names_list = [utils.get_subject_name(sub_id) for sub_id in utils.possible_subjects_id(language)]
subject_ids_list = utils.possible_subjects_id(language)

def dilate_img(img):
    if type(img)==str:
        img = nib.load(img)
    data = img.get_data()
    new_data = dilate_data(data)
    new_img = new_img_like(img, new_data)
    return new_img

def dilate_data(data):
    new_data = np.zeros(data.shape)
    nx, ny, nz = data.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if data[i, j, k] == 0:
                    new_data[i, j, k] = max(
                                        data[max(0, i-1), j, k],
                                        data[min(i+1, nx-1), j, k],
                                        data[i, max(j-1, 0), k],
                                        data[i, min(j+1, ny-1), k],
                                        data[i, j, max(k-1, 0)],
                                        data[i, j, min(k+1, nz-1)],
                                       )
                else:
                    new_data[i, j, k] = data[i, j, k] 
    return new_data

def voxel_masker(coords, img, plot=False, **kwargs_masker):
    """Returns:
        - masker for given coordinates
        - value for input image at this location
        - voxel coordinates
    """
    if type(img)==str:
        img = nib.load(img)
    affine = img.affine[:3, :3]
    translation = img.affine[:3, 3]
    data_coords = np.matmul(np.linalg.inv(affine), np.array(coords) - translation)
    a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords).astype(int)
    value = img.get_data()[a, b, c]
    new_data = np.zeros(img.get_data().shape)
    new_data[a,b,c] = 1
    masker = nilearn.input_data.NiftiMasker(new_img_like(img, new_data))
    masker.set_params(**kwargs_masker)
    masker.fit()
    if plot:
        plotting.plot_glass_brain(masker.mask_img, display_mode='lzry')
        plotting.show()
    return masker, value, [a, b, c]

def extract_voxel_data(masker, coords):
    """Given a voxel coordinate, return the voxel index in the numpy array
    associated with the masker
    """
    data = np.zeros(masker.mask_img.shape)
    a, b, c = coords
    data[a, b, c] = 1
    img = new_img_like(masker.mask_img, data)
    data = masker.transform(img)[0]
    return np.where(data > 0)

def create_one_sample_t_test(
    name, 
    maps, 
    output_dir, 
    smoothing_fwhm=None, 
    vmax=None, 
    design_matrix=None, 
    p_val=0.001, 
    fdr=0.01, 
    loc=0,
    scale=1,
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
    z_th = norm.isf(p_val, loc=loc, scale=scale)  # 3.09

    # apply fdr to zmap
    thresholded_zmap, th = map_threshold(stat_img=z_map,
                                         alpha=fdr,
                                         height_control='fdr',
                                         cluster_threshold=0,
                                         two_sided=False
                                        )
    print(z_th, th)
    # effect size-map
    eff_map = model.compute_contrast(output_type='effect_size')

    thr = np.abs(thresholded_zmap.get_data())
 
    return z_map, eff_map, new_img_like(eff_map, (thr > z_th)), (thr > z_th)

def extract_roi_data(map_, masker, mask_img=None):
    try:
        array = masker.transform(map_)[0]
        if mask_img is not None:
            mask = masker.transform(mask_img)
            mask[np.isnan(mask)] = 0
            mask = mask.astype(int)
            mask = mask[0].astype(bool)
            array = array[mask]
        maximum = np.max(array)
        third_quartile = np.percentile(array, 75)
        median = np.median(array)
        size = len(array.reshape(-1))
    except ValueError as e:
        print(e)
        maximum = np.nan
        third_quartile = np.nan
        median = np.nan
        size = 0
    return median, third_quartile, maximum, size

def linear_fit(x, y, num_points):
    """Fit linear curve between two 2D points."""
    if x[0]==x[1]:
        return np.array([x[0]]*(num_points+2)), np.array([y[0]]*(num_points+2))
    a = (y[1] - y[0])/(x[1] - x[0])
    b = y[0] - a * x[0]
    X = np.linspace(x[0], x[1], num_points + 2)
    Y = np.array([a * item + b for item in X])
    return X, Y

def fit_per_roi(maps, atlas_maps, labels, global_mask, PROJECT_PATH, mask_img=None):
    print("\tLooping through labeled masks...")
    median = np.zeros((len(labels), len(maps)))
    third_quartile = np.zeros((len(labels), len(maps)))
    maximum = np.zeros((len(labels), len(maps)))
    size = np.zeros((len(labels), len(maps)))
    for index_mask in tqdm(range(len(labels))):
        masker = utils.get_roi_mask(atlas_maps, index_mask, labels, global_mask=global_mask, PROJECT_PATH=PROJECT_PATH)
        results = Parallel(n_jobs=-2)(delayed(extract_roi_data)(
            map_, 
            masker, 
            mask_img=mask_img
        ) for map_ in maps)
        results = list(zip(*results))
        median[index_mask, :] = np.hstack(np.array(results[0]))
        third_quartile[index_mask, :] = np.hstack(np.array(results[1]))
        maximum[index_mask, :] = np.hstack(np.array(results[2]))
        size[index_mask, :] = np.hstack(np.array(results[3]))
            
            
    print("\t\t-->Done")
    return median, third_quartile, maximum, size

def save_group_information(
    maps, 
    X,
    labels, 
    model_type, 
    PROJECT_PATH=PROJECT_PATH,
    method='mean',
    factor=0,
    voxel_wise=False,
    plot=False,
    masker=dilated_global_masker_95,
    p_val=0.5,
    fdr=0.01,
    plot_coords=[]
):
    data = []
    all_results = {}
    saving_path = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/median_{}.npy'.format(model_type)
    if os.path.exists(saving_path):
        data = np.load(saving_path)
    else:
        if 'gpt2' in model_type:
            mask_img=os.path.join(ALL_MASKS_PATH, 'gpt2_pre-20_1_norm-inf_norm-inf_{}_hidden-all-layers_pca_300_significant-voxels.nii.gz')
        elif 'lstm' in model_type:
            mask_img=os.path.join(ALL_MASKS_PATH, 'LSTM_embedding-size-600_nhid-300_nlayers-1_dropout-02_memory-size-inf_wiki-kristina_english_{}_hidden-all-layers_pca_300_significant-voxels.nii.gz')
        elif 'bert' in model_type:
            mask_img=os.path.join(ALL_MASKS_PATH, 'bert-base-uncased_pre-2_1_post-0_norm-None_norm-inf_temporal-shifting-0_hidden-all-layers_pca_300_significant-voxels.nii.gz')    
        for index, model in enumerate(maps):
            if voxel_wise:
                results = Parallel(n_jobs=-2)(delayed(masker.transform)(dilate_img(map_)) for map_ in model)
                data.append(np.vstack(results).T)
            else:
                median_, third_quartile_, maximum_, size_ = fit_per_roi(
                    model, 
                    atlas_maps, 
                    labels, 
                    global_mask=None, 
                    mask_img=dilate_img(mask_img),
                    PROJECT_PATH=PROJECT_PATH
                )
                data.append(median_)
        data= np.stack(data, axis=0)
        np.save(saving_path, data)
    
    context_analysis(data, X, labels, model_type, masker=masker, method=method, factor=factor, plot=plot, PROJECT_PATH=PROJECT_PATH, p_val=p_val, fdr=fdr, plot_coords=plot_coords)


def context_analysis(data, X, labels, model_type, masker=dilated_global_masker_95, method='mean', factor=0, plot=False, cst=0, PROJECT_PATH=PROJECT_PATH, fdr=0.01, p_val=0.001, plot_coords=[]):
    """Do regional context analysis.
    """
    model_folder = os.path.join(PROJECT_PATH, 'derivatives/fMRI/context/english/{}_fdr-{}'.format(model_type, fdr))
    utils.check_folder(model_folder)
    tmp_path = os.path.join(model_folder, 'tmp_slopes_{}_fdr-{}.npy'.format(model_type, fdr))
    scaled_data_path = os.path.join(model_folder, 'scaled_data_{}_fdr-{}.npy'.format(model_type, fdr))
    if os.path.exists(tmp_path) and os.path.exists(scaled_data_path):
        tmp = np.load(tmp_path)
        scaled_data = np.load(scaled_data_path)
    else:
        scaled_data = np.stack([StandardScaler(with_mean=True, with_std=True).fit_transform(data[:, index, :]) for index in tqdm(range(data.shape[1]))], axis=0)
        scaled_data = np.rollaxis(scaled_data, 1, 0)
        context_sizes = None

        ### Method with difference
        #abs_dev = mean_abs_deviation(data, axis=0)
        #cst = np.mean(abs_dev)   #np.mean(data[-1, :, :] - data[0, :, :])
        #tmp = data[-1, :, :] - data[0, :, :] - cst  #- factor * abs_dev - cst
        
        ### Method with slope
        def f(X, Y):
            return LinearRegression().fit(X, Y).coef_[0][0]
        tmp = [np.array(
            Parallel(n_jobs=-1)(delayed(f)(
                X.reshape(-1, 1), 
                data[:, index_voxel, subject].reshape(-1, 1)
            ) for subject in range(data.shape[-1]))
            ) for index_voxel in tqdm(range(data.shape[1]))]
        tmp = np.stack(tmp, axis=0)
        np.save(tmp_path, tmp)
        np.save(scaled_data_path, scaled_data)

    tmp = np.rollaxis(tmp, 1, 0)
    tmp = [tmp[i, :] for i in range(tmp.shape[0])]
    imgs = masker.inverse_transform(tmp)

    
    output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format('english', model_type))
    utils.check_folder(output_dir)
    

    z_map, eff_map, img_mask, array_mask = create_one_sample_t_test(model_type, imgs, output_dir=output_dir, fdr=fdr, p_val=p_val, loc=0)

    nib.save(z_map, os.path.join(model_folder, 'zmap_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(eff_map, os.path.join(model_folder, 'eff-map_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(img_mask, os.path.join(model_folder, 'mask_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    np.save(os.path.join(model_folder, 'mask_{}_fdr-{}.npy'.format(model_type, fdr)), array_mask)
    
    significant_values = np.array(masker.transform(img_mask)[0])
    effect_sizes = np.array(masker.transform(eff_map)[0])

    median = np.median(scaled_data, axis=-1)
    lower, upper = bootstrap_confidence_interval(scaled_data, axis=-1, statistic=np.median, confidence=0.95)
    np.save(os.path.join(model_folder, 'lower-conf-interv_{}_fdr-{}.npy'.format(model_type, fdr)), lower)
    np.save(os.path.join(model_folder, 'upper-conf-interv_{}_fdr-{}.npy'.format(model_type, fdr)), upper)
    limit = np.max(lower, axis=0)

    def function(X, m, l, s):
        """Return first index of X at which the limit l if crossed by m, if s (significant.)"""
        try:
            return int(round(X[np.argwhere(m > l)[0]][0], 0)) if s else None
        except:
            return 1 if s else None

    context_sizes = Parallel(n_jobs=-1)(delayed(function)(X, median[:, index], limit[index], True) for index in range(median.shape[-1]))

    new_X = [np.array([
        X[list(X).index(context_sizes[index]) - 1] if list(X).index(context_sizes[index]) > 0 else X[list(X).index(context_sizes[index])], 
        context_sizes[index]
    ]) for index in range(len(context_sizes))]
    estimations = Parallel(n_jobs=-2)(delayed(linear_fit)(
        new_X[index], 
        median[list(X).index(new_X[index][0]): list(X).index(new_X[index][1]) + 1, index], 
        num_points=10
    ) for index in range(len(context_sizes)))

    estimated_X = np.stack(np.array(list(zip(*estimations))[0]), axis=-1)
    estimated_median = np.stack(np.array(list(zip(*estimations))[1]), axis=-1)
    context_sizes = Parallel(n_jobs=-1)(delayed(function)(estimated_X[:, index], estimated_median[:, index], limit[index], True) for index in range(median.shape[-1]))
    np.save(os.path.join(model_folder, 'context_sizes_{}_fdr-{}.npy'.format(model_type, fdr)), np.array(context_sizes))


def mean_abs_deviation(array, axis):
    return np.mean(np.abs(array - np.mean(array, axis=axis)), axis=axis)

def std_err_median(array, axis, ddof=1):
    n = array.shape[-1]
    return np.sqrt(np.sum(np.abs(array - np.median(array, axis=axis))**2, axis=axis)/(n-ddof))/np.sqrt(n)

def mean_confidence_interval(data, confidence=0.95): 
    """data shape:  #data-points * #subjects
    """
    a = 1.0 * np.array(data) 
    n = a.shape[-1] 
    m, se = np.mean(a, axis=-1), std_err_median(a, axis=-1) 
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., [n-1] * a.shape[0]) 
    return m, m-h, m+h 

def median_confidence_interval(data, confidence=0.95): 
    """data shape:  #data-points * #subjects
    """
    a = 1.0 * np.array(data) 
    n = a.shape[-1] 
    m, se = np.median(a, axis=-1), scipy.stats.sem(a, axis=-1) 
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., [n-1] * a.shape[0]) 
    return m, m-h, m+h 

def bootstrap_confidence_interval(array, axis, statistic=np.median, confidence=0.95, seed=1111):
    np.random.seed(seed)
    # bootstrap
    scores = list()
    for _ in range(100):
        # bootstrap sample
        indices = np.random.randint(0, 51, 51)
        if len(array.shape)==2:
            sample = array[:, indices]
        elif len(array.shape)==3:
            sample = array[:, :, indices]
        # calculate and store statistic
        statistic_results = statistic(sample, axis=axis)
        scores.append(statistic_results)
    scores = np.stack(scores, axis=0)
    # calculate confidence intervals
    alpha = 100 - confidence*100
    # calculate lower percentile 
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = np.percentile(scores, lower_p, axis=0)
    # calculate upper percentile
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = np.percentile(scores, upper_p, axis=0)
    return lower, upper

def ROI_voxels_indexes(img, masker, labels=labels, atlas_maps=atlas_maps, PROJECT_PATH=PROJECT_PATH):
    data = masker.transform(img)
    data_indexes = np.arange(1, 1 + data.shape[-1]).reshape(1, -1)
    img_indexes = masker.inverse_transform(data_indexes)
    
    def f(index):
        masker = utils.get_roi_mask(atlas_maps, index, labels, global_mask=None, PROJECT_PATH=PROJECT_PATH)
        result = masker.transform(img_indexes)
        return result[result>0]
        
    results = Parallel(n_jobs=-2)(delayed(f)(
        index_mask,
    ) for index_mask, label in enumerate(labels))
    
    result = {}
    for index_label, label in enumerate(labels):
        result[label] = np.array([i for i in results[index_label]])
        
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Context effect analysis.""")
    parser.add_argument("--model_type", type=str, default='old_gpt2_voxel-wise_pre-20', help="Name of the model.")
    parser.add_argument("--layer", type=str, help="Layer in the model.")
    parser.add_argument("--fdr", type=float, default=0.01, help="FDR value for correction")
    parser.add_argument("--pval", type=float, default=0.5, help="p-value threshold if needed.")

    args = parser.parse_args()

    model_names = [
        'old_gpt2_pre-20_token-1_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-2_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-5_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-10_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-15_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-20_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-30_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-40_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-60_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-80_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
        'old_gpt2_pre-20_token-100_norm-inf_norm-inf_temporal-shifting-0_unk_hidden-layer-{layer}'.format(layer=args.layer).replace('unk', '{}'),
    ]

    data_full = reporting.get_model_data(model_names, language, OUTPUT_PATH)
    maps = [data_full[name]['Pearson_coeff'] for name in data_full.keys()]

    X = np.array([1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100])

    save_group_information(
                            maps, 
                            X, 
                            ['voxel-{}'.format(i) for i in range(26074)], 
                            masker=dilated_global_masker_95,
                            model_type=args.model_type + '_layer-{}'.format(args.layer), 
                            PROJECT_PATH=PROJECT_PATH,
                            method='mean',
                            plot=False,
                            voxel_wise=True,
                            factor=0,
                            fdr=args.fdr,
                            p_val=args.pval,
    )