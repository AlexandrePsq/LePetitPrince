import os
import gc
import glob
import itertools
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed

import scipy
import nistats
import numpy as np
import pandas as pd
import argparse
#import pingouin as pg
from scipy.stats import norm
import statsmodels.api as sm 
from sklearn import manifold
from scipy.special import erf
from scipy.optimize import curve_fit
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.manifold import LocallyLinearEmbedding
from statsmodels.stats.multitest import multipletests
from nistats.second_level_model import SecondLevelModel
from sklearn.cluster import AgglomerativeClustering, KMeans

import matplotlib
import seaborn as sns
import matplotlib.cm as cmx
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colorbar import ColorbarBase, make_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import nibabel as nib
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img, new_img_like
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from nilearn._utils.niimg import _safe_get_data
from nilearn import plotting
from nilearn import datasets
from nilearn.surface import vol_to_surf

import utils
import reporting
from logger import Logger

#### Variables ####
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

plot_coords = [
    [-48, 15, -27],
    [-54, -12, -12],
    [-51, -39, 3],
    [-39, -57, 18],
    [-45, 33, -6],
    [-51, 21, 21],
    [-42, 10, 22],
    [-35, -10, 65],
    [-61, -33, -6],
    [-17, 30, 61],
    [-43, 20, 49],
    [-7, 0, 50],
    [-23, 31, 48],
    [-64, -42, 38],
    [56, -41, 14],
    [-26, -97, 5],
    [-61, -11, -6],
    [56, -41, -14],
    [61, -22, -6],
    [-61, -33, -7],
    [-17, 30, 61],
    [-43, 20, 49],
    [-7, 0, 50],
    [-64, -42, 38],
    [-4, 46, 12],
    [4, 0, 47],
]

### Atlas definiton
#atlas_maps, labels = reporting.load_atlas()
atlas_detailed = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)

labels = [roi.decode("utf-8") for roi in atlas_detailed['labels']]
atlas_maps = nilearn.image.load_img(atlas_detailed['maps'])


### Masker definition
global_masker_50 = reporting.fetch_masker(f"{ALL_MASKS_PATH}/global_masker_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)
global_masker_95 = reporting.fetch_masker(f"{ALL_MASKS_PATH}/global_masker_95%_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)


original_masker = global_masker_50
new_masker = global_masker_95
original_masker.set_params(detrend=False, standardize=False)
new_masker.set_params(detrend=False, standardize=False)


params = global_masker_95.get_params()
new_img = new_img_like(global_masker_95.mask_img, scipy.ndimage.binary_dilation(global_masker_95.mask_img.get_fdata()))
dilated_global_masker_95 = NiftiMasker()
dilated_global_masker_95.set_params(**params)
dilated_global_masker_95.mask_img = new_img
dilated_global_masker_95.fit()


#### Functions ####

def extract_roi_data(map_, masker, mask_img=None, threshold_img=None, voxels_filter=None, threshold_error=10**8):
    try:
        array = masker.transform(map_)[0]
        if mask_img is not None:
            mask = masker.transform(mask_img)
            mask[np.isnan(mask)] = 0
            mask = mask.astype(int)
            mask = mask[0].astype(bool)
            array = array[mask]
            if threshold_img is not None:
                threshold = masker.transform(threshold_img)
                threshold = threshold[mask]
                if voxels_filter is not None:
                    mask2 = threshold >= voxels_filter
                    array = array[mask2]
                    threshold = threshold[mask2]
                threshold[threshold==0.0] = threshold_error # due to some voxel equal to 0 in threshold image we have nan value
                array = np.divide(array, threshold)
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

def fit_per_roi(
    maps, 
    atlas_maps, 
    labels, 
    global_mask, 
    PROJECT_PATH, 
    mask_img=None, 
    threshold_img=None, 
    voxels_filter=None, 
    threshold_error=10**8):
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
            mask_img=mask_img,
            threshold_img=threshold_img, 
            voxels_filter=voxels_filter, 
            threshold_error=threshold_error
        ) for map_ in maps)
        results = list(zip(*results))
        median[index_mask, :] = np.hstack(np.array(results[0]))
        third_quartile[index_mask, :] = np.hstack(np.array(results[1]))
        maximum[index_mask, :] = np.hstack(np.array(results[2]))
        size[index_mask, :] = np.hstack(np.array(results[3]))
            
            
    print("\t\t-->Done")
    return median, third_quartile, maximum, size

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
    value = img.get_fdata()[a, b, c]
    new_data = np.zeros(img.get_fdata().shape)
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
    fwhm=6,
    logs_path=None
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
    z_th = norm.isf(p_val, loc=loc, scale=scale)  # 3.09

    # apply fdr to zmap
    thresholded_zmap, th = map_threshold(stat_img=z_map,
                                         alpha=fdr,
                                         height_control='fdr',
                                         cluster_threshold=0,
                                         two_sided=False
                                        )
    if logs_path is not None:
        utils.write(logs_path, str(z_th) + ' - ' +  str(th))
    # effect size-map
    eff_map = model.compute_contrast(output_type='effect_size')

    thr = np.abs(thresholded_zmap.get_fdata())
 
    return z_map, eff_map, new_img_like(eff_map, (thr > z_th)), (thr > z_th)

def dilate_img(img):
    if type(img)==str:
        img = nib.load(img)
    data = img.get_fdata()
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
def bootstrap_sampling(array, axis, n_times, sample_size, seed=1111):
    np.random.seed(seed)
    max_value = array.shape[axis]
    # bootstrap
    samples = list()
    for _ in range(n_times):
        # bootstrap sample
        indices = np.random.randint(0, max_value, sample_size)
        samples.append(np.take(array, indices, axis))
    return samples

def get_group_information(
    maps, 
    X,
    labels, 
    model_type, 
    PROJECT_PATH=PROJECT_PATH,
    atlas_maps=atlas_maps,
    method='mean',
    factor=0,
    voxel_wise=False,
    plot=False,
    masker=dilated_global_masker_95,
    p_val=0.001,
    fdr=0.01,
    plot_coords=[],
    bootstrap=True,
    n_times=100, 
    sample_size=50,
):
    data = []
    saving_path = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/median_{}_fdr-{}.npy'.format(model_type, fdr)
    model_folder = os.path.join(PROJECT_PATH, 'derivatives/fMRI/context/english/{}_fdr-{}_pval-{}'.format(model_type, fdr, p_val))
    utils.check_folder(model_folder)
    logs_path = os.path.join(model_folder, 'logs.txt')
    if os.path.exists(saving_path):
        utils.write(logs_path, 'Loading data...')
        data = np.load(saving_path)
        utils.write(logs_path, 'Data loaded.')
    else:
        utils.write(logs_path, 'Computing data...')
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
                    mask_img=None, #dilate_img(mask_img) if mask_img is not None else 
                    PROJECT_PATH=PROJECT_PATH
                )
                data.append(median_)
        utils.write(logs_path, 'Data computed.')
        data= np.stack(data, axis=0)
        np.save(saving_path, data)
    
    if bootstrap:
        context_sizes, z_map, eff_map, img_mask, array_mask = context_analysis_bootstrapped(data, X, labels, model_type, atlas_maps=atlas_maps, n_times=n_times, sample_size=sample_size, voxel_wise=voxel_wise, masker=masker, method=method, factor=factor, plot=plot, titles=['Signal per subject', 'Scaled signal per subject'], PROJECT_PATH=PROJECT_PATH, p_val=p_val, fdr=fdr, plot_coords=plot_coords)
    else:
        context_sizes, z_map, eff_map, img_mask, array_mask = context_analysis(data, X, labels, model_type, atlas_maps=atlas_maps, voxel_wise=voxel_wise, masker=masker, method=method, factor=factor, plot=plot, titles=['Signal per subject with {}'.format(model_type), 'Scaled signal per subject with {}'.format(model_type)], PROJECT_PATH=PROJECT_PATH, p_val=p_val, fdr=fdr, plot_coords=plot_coords)
    
    return context_sizes, z_map, eff_map, img_mask, array_mask

def context_analysis_bootstrapped(data, X, labels, model_type, atlas_maps=atlas_maps, voxel_wise=False, masker=dilated_global_masker_95, method='mean', n_times=100, sample_size=50, factor=0, plot=False, cst=0, titles=['', ''], PROJECT_PATH=PROJECT_PATH, fdr=0.01, p_val=0.001, plot_coords=plot_coords):
    """Do regional context analysis.
    """
    model_folder = os.path.join(PROJECT_PATH, 'derivatives/fMRI/context/english/{}_fdr-{}_pval-{}'.format(model_type, fdr, p_val))
    logs_path = os.path.join(model_folder, 'logs.txt')
    utils.check_folder(model_folder)
    tmp_path = os.path.join(model_folder, 'tmp_slopes_{}_fdr-{}.npy'.format(model_type, fdr))
    scaled_data_path = os.path.join(model_folder, 'scaled_data_{}_fdr-{}.npy'.format(model_type, fdr))
    data_path = os.path.join(model_folder, 'data_{}_fdr-{}.npy'.format(model_type, fdr))
    np.save(data_path, data) # size: (#models, #voxels, #subjects)
    dimensions = atlas_maps.shape
    if os.path.exists(tmp_path) and os.path.exists(scaled_data_path):
        utils.write(logs_path, 'Resuming operations...')
        tmp = np.load(tmp_path)
        scaled_data = np.load(scaled_data_path)
        utils.write(logs_path, 'Slopes & Scaled-data loaded.')
    else:
        utils.write(logs_path, 'Computing: Scaled-data & Slopes...')
        scaled_data = np.stack([StandardScaler(with_mean=True, with_std=True).fit_transform(data[:, index, :]) for index in tqdm(range(data.shape[1]))], axis=0)
        scaled_data = np.rollaxis(scaled_data, 1, 0)
        context_sizes = None

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
        np.save(scaled_data_path, scaled_data) # size: (#models, #voxels, #subjects)
    utils.write(logs_path, 'Computed.')
    
    if voxel_wise:
        tmp = np.rollaxis(tmp, 1, 0) # from (#voxels, #subjects) to (#subjects, #voxels)

        tmp = [tmp[i, :] for i in range(tmp.shape[0])]
        imgs = masker.inverse_transform(tmp)

        output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format('english', model_type))
        utils.check_folder(output_dir)

        utils.write(logs_path, 'Computing maps...')
        z_map, eff_map, img_mask, array_mask = create_one_sample_t_test(model_type, imgs, output_dir=output_dir, fdr=fdr, p_val=p_val, loc=0, logs_path=logs_path)

        significant_values = np.array(masker.transform(img_mask)[0])
        effect_sizes = np.array(masker.transform(eff_map)[0])
    
    else:
        tstat, pvalue = scipy.stats.ttest_1samp(tmp, 0, axis=1, alternative='greater')
        tstat[np.isnan(tstat)] = 0
        pvalue[np.isnan(pvalue)] = 1
        z_values = norm.isf(pvalue)
        significant_values, corrected_pvalues = fdrcorrection(pvalue, alpha=fdr, method='indep', is_sorted=False)
        np.save(os.path.join(model_folder, 'significant-values_{}_fdr-{}.npy'.format(model_type, fdr)), significant_values)
        np.save(os.path.join(model_folder, 'corrected-pvalues_{}_fdr-{}.npy'.format(model_type, fdr)), corrected_pvalues)
        img_mask = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))
        z_map = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))        
        eff_map = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))        
        
        for index_mask, value in tqdm(enumerate(significant_values)):
            value = int(value)
            if value: #(index_mask != 0) and
                mask_ = math_img('img=={}'.format(index_mask+1), img=atlas_maps)
                # mask img
                img_mask = math_img(f"img1 + img2*{index_mask+1}", img1=img_mask, img2=mask_)
                # Z map
                mask_z = math_img('img * {}'.format(z_values[index_mask]), img=mask_)
                z_map = math_img("img1 + img2", img1=z_map, img2=mask_z)
                # eff map
                mask_eff = math_img('img * {}'.format(tstat[index_mask]), img=mask_)
                eff_map = math_img("img1 + img2", img1=eff_map, img2=mask_eff)
        array_mask = img_mask.get_fdata()
    
    nib.save(z_map, os.path.join(model_folder, 'zmap_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(eff_map, os.path.join(model_folder, 'eff-map_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(img_mask, os.path.join(model_folder, 'mask_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    np.save(os.path.join(model_folder, 'mask_{}_fdr-{}.npy'.format(model_type, fdr)), array_mask)
    utils.write(logs_path, 'Bootstrapping...')
    # bootstrap over subject to get statistics
    samples = bootstrap_sampling(scaled_data, -1, n_times=n_times, sample_size=sample_size, seed=1111) # size: #samples* ( #models, #voxels, #subjects)
    median_curves = [np.median(scaled_data_sample, axis=-1) for scaled_data_sample in samples]

    median_all = np.stack(median_curves, axis=0) # size: (#samples, #models, #voxels)
    #std = np.std(median_all, axis=0) # size: (#models, #voxels)
    #limit_curves = [np.max(median - std, axis=0) for median in median_curves] # size: #voxels

    bootstrap = Parallel(n_jobs=-1)(delayed(bootstrap_confidence_interval)(sample, axis=-1, statistic=np.median, confidence=0.95) for sample in samples)
    
    lower, upper = zip(*bootstrap)
    lower = np.stack(lower, axis=0) # size: (#samples, #models, #voxels)
    upper = np.stack(upper, axis=0)
    np.save(os.path.join(model_folder, 'lower-conf-interv_{}_fdr-{}.npy'.format(model_type, fdr)), lower)
    np.save(os.path.join(model_folder, 'upper-conf-interv_{}_fdr-{}.npy'.format(model_type, fdr)), upper)
    limit = np.max(lower, axis=1) # size: (#samples, #models, #voxels)
    
    def function(X, m, l, s):
        """Return first index of X at which the limit l if crossed by m, if s (significant.)"""
        try:
            return int(round(X[np.argwhere(m > l)[0]][0], 0)) if s else None
        except:
            return min(X) if s else None
    
    def function2(X, context_sizes, median, limit, num_points=10):
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
        return context_sizes

    utils.write(logs_path, 'Computing context sizes...')
    context_sizes_all = []
    for sample_index in tqdm(range(len(samples))):
        context_sizes = Parallel(n_jobs=-1)(delayed(function)(X, median_curves[sample_index][:, index], limit[sample_index, :][index], True) for index in range(median_all.shape[-1]))
        context_sizes = function2(X, context_sizes, median_curves[sample_index], limit[sample_index, :], num_points=10)
        context_sizes_all.append(np.array(context_sizes))

    np.save(os.path.join(model_folder, 'context_sizes_bootstrap_{}_fdr-{}.npy'.format(model_type, fdr)), np.stack(context_sizes_all, axis=0))
    
    if plot:
        for index_label, label in tqdm(enumerate(labels)):
            fig1 = plt.figure(figsize=(20, 10))
            gs = fig1.add_gridspec(2, 2, height_ratios=[10, 90])

            ax1 = fig1.add_subplot(gs[0, :])
            ax2 = fig1.add_subplot(gs[1, 0])
            ax3 = fig1.add_subplot(gs[1, 1])
            ax1.axis('off')

            ax1.text(0, 1, 'Effect size: {} \np-value: {}'.format(round(effect_sizes[index_label], 2), significant_values[index_label]))
            #ax3.fill_between(X, lower, upper, color='red', alpha=0.7)

            format_ax(
                ax2, 
                X, 
                data[:, index_label, :], 
                titles[0], 
                'Context size', 
                'R value',
                bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10}
            )
            format_ax(
                ax3, 
                X, 
                scaled_data[:, index_label, :], 
                titles[1], 
                'Context size', 
                'Scaled R value',
                vertical_line=context_sizes[index_label]
            )
            title_color = 'green' if significant_values[index_label] else 'red'
            plt.suptitle(label + ' - ' + model_type, fontsize=40, bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10})
            plt.subplots_adjust(top=2, bottom=0.1, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
            plt.tight_layout()
            plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/{}_{}_method-{}_significant-factor-{}.png'.format(model_type, label.replace(' ', '-'), method, factor))
            plt.show()
            if input()!='':
                break
    else:
        for coords in plot_coords:
            label = '{}_{}_{}'.format(*coords)
            kwargs_masker = {'detrend': False,
                 'dtype': None,
                 'high_pass': None,
                 'low_pass': None,
                 'memory_level': 0,
                 'smoothing_fwhm': None,
                 'standardize': False,
                 't_r': None,
                 'verbose': 0
                }
            _, _, data_coords = voxel_masker(coords, masker.mask_img, plot=False, **kwargs_masker)
            index = extract_voxel_data(masker, data_coords)[0][0]
            title_color = 'green' if significant_values[index] else 'red'
            
            fig1 = plt.figure(figsize=(20, 5))
            gs = fig1.add_gridspec(2, 2, height_ratios=[10, 90])

            ax1 = fig1.add_subplot(gs[0, :])
            ax2 = fig1.add_subplot(gs[1, 0])
            ax3 = fig1.add_subplot(gs[1, 1])
            ax1.axis('off')

            ax1.text(0, 1, 'Effect size: {} \np-value: {}'.format(round(effect_sizes[index], 2), significant_values[index]))
            #ax3.fill_between(X, lower, upper, color='red', alpha=0.7)

            format_ax(
                ax2, 
                X, 
                data[:, index, :], 
                titles[0], 
                'Context size', 
                'R value',
                bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10}
            )
            format_ax(
                ax3, 
                X, 
                scaled_data[:, index, :], 
                titles[1], 
                'Context size', 
                'Scaled R value',
                vertical_line=np.mean(np.stack(context_sizes_all, axis=0), axis=0)[index]
            )
            #plt.suptitle(label + ' - ' + model_type, fontsize=40, bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10})
            plt.subplots_adjust(top=2, bottom=0.1, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
            plt.tight_layout()
            saving_path = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/{}_{}_method-{}_significant-factor-{}.png'.format(model_type, label.replace(' ', '-'), method, factor)
            plt.savefig(saving_path, bbox_inches = 'tight', pad_inches = 0)
            plt.show()
    return context_sizes, z_map, eff_map, img_mask, array_mask


def context_analysis(data, X, labels, model_type, atlas_maps=atlas_maps, voxel_wise=False, masker=dilated_global_masker_95, method='mean', factor=0, plot=False, cst=0, titles=['', ''], PROJECT_PATH=PROJECT_PATH, fdr=0.01, p_val=0.001, plot_coords=[]):
    """Do regional context analysis.
    """
    model_folder = os.path.join(PROJECT_PATH, 'derivatives/fMRI/context/english/{}_fdr-{}_pval-{}'.format(model_type, fdr, p_val))
    logs_path = os.path.join(model_folder, 'logs.txt')
    utils.check_folder(model_folder)
    tmp_path = os.path.join(model_folder, 'tmp_slopes_{}_fdr-{}.npy'.format(model_type, fdr))
    scaled_data_path = os.path.join(model_folder, 'scaled_data_{}_fdr-{}.npy'.format(model_type, fdr))
    dimensions = atlas_maps.shape
    if os.path.exists(tmp_path) and os.path.exists(scaled_data_path):
        utils.write(logs_path, 'Resuming operations...')
        tmp = np.load(tmp_path)
        scaled_data = np.load(scaled_data_path)
        utils.write(logs_path, 'Slopes & Scaled-data loaded.')
    else:
        utils.write(logs_path, 'Computing: Scaled-data & Slopes...')
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
    utils.write(logs_path, 'Computed.')
    
    if voxel_wise:
        tmp = np.rollaxis(tmp, 1, 0)
        tmp = [tmp[i, :] for i in range(tmp.shape[0])]
        imgs = masker.inverse_transform(tmp)

        output_dir = os.path.join(PROJECT_PATH, 'derivatives/fMRI/analysis/{}/{}'.format('english', model_type))
        utils.check_folder(output_dir)

        utils.write(logs_path, 'Computing maps...')
        z_map, eff_map, img_mask, array_mask = create_one_sample_t_test(model_type, imgs, output_dir=output_dir, fdr=fdr, p_val=p_val, loc=0, logs_path=logs_path)

        significant_values = np.array(masker.transform(img_mask)[0])
        effect_sizes = np.array(masker.transform(eff_map)[0])
    
    else:
        tstat, pvalue = scipy.stats.ttest_1samp(tmp, 0, axis=1, alternative='greater')
        tstat[np.isnan(tstat)] = 0
        pvalue[np.isnan(pvalue)] = 1
        z_values = norm.isf(pvalue)
        significant_values, corrected_pvalues = fdrcorrection(pvalue, alpha=fdr, method='indep', is_sorted=False)
        np.save(os.path.join(model_folder, 'significant-values_{}_fdr-{}.npy'.format(model_type, fdr)), significant_values)
        np.save(os.path.join(model_folder, 'corrected-pvalues_{}_fdr-{}.npy'.format(model_type, fdr)), corrected_pvalues)
        img_mask = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))
        z_map = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))        
        eff_map = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((dimensions[1], dimensions[2])) for i in range(dimensions[0])], axis=0))
        
        for index_mask, value in tqdm(enumerate(significant_values)):
            value = int(value)
            if value:
                mask_ = math_img('img=={}'.format(index_mask+1), img=atlas_maps)
                # mask img
                img_mask = math_img("img1 + img2", img1=img_mask, img2=mask_)
                # Z map
                mask_z = math_img('img * {}'.format(z_values[index_mask]), img=mask_)
                z_map = math_img("img1 + img2", img1=z_map, img2=mask_z)
                # eff map
                mask_eff = math_img('img * {}'.format(tstat[index_mask]), img=mask_)
                eff_map = math_img("img1 + img2", img1=eff_map, img2=mask_eff)
        array_mask = img_mask.get_fdata()
    
    nib.save(z_map, os.path.join(model_folder, 'zmap_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(eff_map, os.path.join(model_folder, 'eff-map_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    nib.save(img_mask, os.path.join(model_folder, 'mask_{}_fdr-{}.nii.gz'.format(model_type, fdr)))
    np.save(os.path.join(model_folder, 'mask_{}_fdr-{}.npy'.format(model_type, fdr)), array_mask)
    utils.write(logs_path, 'Bootstrapping...')
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
            return min(X) if s else None

    utils.write(logs_path, 'Computing context sizes...')
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
    
    if plot:
        for index_label, label in tqdm(enumerate(labels)):
            fig1 = plt.figure(figsize=(20, 10))
            gs = fig1.add_gridspec(2, 2, height_ratios=[10, 90])

            ax1 = fig1.add_subplot(gs[0, :])
            ax2 = fig1.add_subplot(gs[1, 0])
            ax3 = fig1.add_subplot(gs[1, 1])
            ax1.axis('off')

            ax1.text(0, 1, 'Effect size: {} \np-value: {}'.format(round(effect_sizes[index_label], 2), significant_values[index_label]))
            #ax3.fill_between(X, lower, upper, color='red', alpha=0.7)

            format_ax(
                ax2, 
                X, 
                data[:, index_label, :], 
                titles[0], 
                'Number of tokens', 
                'R value'
            )
            format_ax(
                ax3, 
                X, 
                scaled_data[:, index_label, :], 
                titles[1], 
                'Number of tokens', 
                'Scaled R value',
                vertical_line=context_sizes[index_label]
            )
            title_color = 'green' if significant_values[index_label] else 'red'
            plt.suptitle(label + ' - ' + model_type, fontsize=40, bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10})
            plt.subplots_adjust(top=2, bottom=0.1, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
            plt.tight_layout()
            plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/{}_{}_method-{}_significant-factor-{}.png'.format(model_type, label.replace(' ', '-'), method, factor))
            plt.show()
            if input()!='':
                break
    else:
        for coords in plot_coords:
            label = '{}_{}_{}'.format(*coords)
            kwargs_masker = {'detrend': False,
                 'dtype': None,
                 'high_pass': None,
                 'low_pass': None,
                 'memory_level': 0,
                 'smoothing_fwhm': None,
                 'standardize': False,
                 't_r': None,
                 'verbose': 0
                }
            _, _, data_coords = voxel_masker(coords, masker.mask_img, plot=False, **kwargs_masker)
            index = extract_voxel_data(masker, data_coords)[0][0]
            
            fig1 = plt.figure(figsize=(20, 10))
            gs = fig1.add_gridspec(2, 2, height_ratios=[10, 90])

            ax1 = fig1.add_subplot(gs[0, :])
            ax2 = fig1.add_subplot(gs[1, 0])
            ax3 = fig1.add_subplot(gs[1, 1])
            ax1.axis('off')

            ax1.text(0, 1, 'Effect size: {} \np-value: {}'.format(round(effect_sizes[index], 2), significant_values[index]))
            #ax3.fill_between(X, lower, upper, color='red', alpha=0.7)

            format_ax(
                ax2, 
                X, 
                data[:, index, :], 
                titles[0], 
                'Number of tokens', 
                'R value'
            )
            format_ax(
                ax3, 
                X, 
                scaled_data[:, index, :], 
                titles[1], 
                'Number of tokens', 
                'Scaled R value',
                vertical_line=context_sizes[index]
            )
            title_color = 'green' if significant_values[index] else 'red'
            plt.suptitle(label + ' - ' + model_type, fontsize=40, bbox={'facecolor': title_color, 'alpha': 0.5, 'pad': 10})
            plt.subplots_adjust(top=2, bottom=0.1, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
            plt.tight_layout()
            plt.savefig('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/{}_{}_method-{}_significant-factor-{}.png'.format(model_type, label.replace(' ', '-'), method, factor))
            plt.show()
    return context_sizes, z_map, eff_map, img_mask, array_mask

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

def linear_fit(x, y, num_points):
    """Fit linear curve between two 2D points."""
    if x[0]==x[1]:
        return np.array([x[0]]*(num_points+2)), np.array([y[0]]*(num_points+2))
    a = (y[1] - y[0])/(x[1] - x[0])
    b = y[0] - a * x[0]
    X = np.linspace(x[0], x[1], num_points + 2)
    Y = np.array([a * item + b for item in X])
    return X, Y

def format_ax(ax, X, data, title, xlabel, ylabel, confidence_interval=95, vertical_line=None, bbox=None):
    """Format a matplotlib ax with given data and information...
    """
    dataframe = pd.DataFrame()
    dataframe['X'] = [int(x) for x in X]
    dataframe = pd.concat([dataframe, pd.DataFrame(data)], axis=1)
    dataframe = pd.melt(dataframe, id_vars=['X'])
    dataframe.columns = ['X', 'subject_id', 'Y']
    plot_ = sns.lineplot(data=dataframe, x='X', y='Y', ci=confidence_interval, ax=ax, estimator=np.median, legend='brief')
    ax.plot(X, data, c='black', alpha=0.2)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(title, fontsize=30, bbox=bbox)
    ax.tick_params(axis='x', labelsize=15, rotation=0)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='y')
    ax.grid(which='major', linestyle=':', linewidth='0.5', color='black', alpha=0.4, axis='x')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='y')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.1, axis='x')
    if (vertical_line!=None): ax.axvline(x=vertical_line ,color='red', linestyle='--')
    legend_elements = [
        Line2D([0], [0], color='black', alpha=0.2, lw=2),
        Line2D([0], [0], color='blue', lw=2, alpha=0.6),
        mpatches.Patch(color='blue', alpha=0.2)
    ]
    legend_names = [
        "Subject's data",
        'Median signal',
        'Confidence interval: {}%'.format(confidence_interval)
    ]
    if (vertical_line!=None):
        legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--'))
        legend_names.append('Context saturation: {}'.format(vertical_line))

    ax.legend(legend_elements, legend_names)


def plot_roi_analysis(model_name, values, labels, plot_name, mask_img=None, saving_path='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/'):
    """Plot at ROI level.
    """
    filtered_sizes = [item for item in values if item is not None]
    new_img_diff = nilearn.image.new_img_like(atlas_maps, np.stack([np.zeros((109, 91)) for i in range(91)], axis=0))
    for index_mask in tqdm(range(len(labels))):
        value = values[index_mask] if values[index_mask] is not None else 0
        if index_mask == 0:
            value = 0

        mask = math_img('img=={}'.format(index_mask), img=atlas_maps)

        mask_ = math_img('img * {}'.format(value), img=mask)
        new_img_diff = math_img("img1 + img2", img1=new_img_diff, img2=mask_)

    if 'gpt2' in model_name:
        mask_img=os.path.join(ALL_MASKS_PATH, 'gpt2_pre-20_1_norm-inf_norm-inf_{}_hidden-all-layers_pca_300_significant-voxels.nii.gz')
    elif 'lstm' in model_name:
        mask_img=os.path.join(ALL_MASKS_PATH, 'LSTM_embedding-size-600_nhid-300_nlayers-1_dropout-02_memory-size-inf_wiki-kristina_english_{}_hidden-all-layers_pca_300_significant-voxels.nii.gz')
    elif 'bert' in model_name:
        mask_img=os.path.join(ALL_MASKS_PATH, 'bert-base-uncased_pre-2_1_post-0_norm-None_norm-inf_temporal-shifting-0_hidden-all-layers_pca_300_significant-voxels.nii.gz')    

    plot_name = '_'.join([plot_name + model_name])
    multi_plot(
        data=new_img_diff, 
        mask=mask_img,
        plot_name=plot_name, 
        saving_path=saving_path,
        return_plot=True, 
        inflated=False,
        suptitle=None, #'Significant Context effect in ROIs (in number of tokens)',
        vmax=np.max(filtered_sizes),
        vmin=0,
        cmap='cold_hot',
        colorbar=False,
        **kwargs)
    plt.close('all')
    fig = plt.figure(figsize=(1,10))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    _draw_colorbar(ax, vmax=np.max(filtered_sizes), vmin=0, cmap='black_red')
    #plt.savefig(os.path.join(saving_path, plot_name + '_colorbar.png'), bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def plot_voxel_wise_analysis(model_name, masker, data, labels, plot_name, vmax=None, save_independently=False, saving_path='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/context/english/raw_figures/', mask_img=None, inflated=False, categorical_values=False, cmap='cold_hot', draw_cbar=True):
    """Plot voxel-wise.
    """
    img = masker.inverse_transform(np.array(data).reshape(1, -1))
    if vmax is None: 
        vmax = np.max(data)
    
    plot_name = '_'.join([plot_name + model_name])
    multi_plot(
        data=img, 
        mask=mask_img,
        plot_name=plot_name, 
        saving_path=saving_path,
        return_plot=True, 
        inflated=inflated,
        suptitle=None, #'Significant Context effect in ROIs (in number of tokens)',
        vmax=vmax,
        vmin=0,
        cmap=cmap,
        colorbar=False,
        save_independently=save_independently,
        categorical_values=categorical_values,
        **kwargs)
    plt.close('all')
    if draw_cbar:
        if categorical_values:
            cmap=cmap
        else:
            cmap='black_red'
        fig = plt.figure(figsize=(1,10))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        _draw_colorbar(ax, vmax=vmax, vmin=0, cmap=cmap)
        if saving_path is not None:
            plt.savefig(os.path.join(saving_path, plot_name + '_colorbar.png'), bbox_inches = 'tight', pad_inches = 0, format='png')
        plt.show()

def bootstrap_confidence_interval(array, axis, statistic=np.median, confidence=0.95, seed=1111):
    np.random.seed(seed)
    withdrawal = array.shape[-1]
    # bootstrap
    scores = list()
    for _ in range(100):
        # bootstrap sample
        indices = np.random.randint(0, withdrawal, withdrawal)
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


#### Preprocessing ####


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute context analysis')
    parser.add_argument("--yaml_path", type=str)

    args = parser.parse_args()

    parameters = utils.read_yaml(args.yaml_path)

    model_names = parameters['model_names']
    X = np.array(parameters['X'])
    fdr = parameters['fdr']
    p_val = parameters['p_val']
    model_type = parameters['model_type']
    bootstrap = parameters['bootstrap']

    data_full = reporting.get_model_data(model_names, language, OUTPUT_PATH=OUTPUT_PATH)
    maps = [data_full[name]['Pearson_coeff'] for name in data_full.keys()]

    # Per Voxels
    #tmp = get_group_information(
    #        maps, 
    #        X, 
    #        ['voxel-{}'.format(i) for i in range(26074)], 
    #        masker=dilated_global_masker_95,
    #        model_type=model_type.format('voxel-wise'), 
    #        PROJECT_PATH=PROJECT_PATH,
    #        method='mean',
    #        plot=False,
    #        voxel_wise=True,
    #        factor=0,
    #        fdr=fdr,
    #        p_val=p_val,
    #        bootstrap=bootstrap,
    #        n_times=100, 
    #        sample_size=50,
    #        plot_coords=plot_coords,
    #    )
    #
    # Per ROI
    tmp = get_group_information(
            maps, 
            X, 
            labels=labels, 
            atlas_maps=atlas_maps,
            masker=dilated_global_masker_95,
            model_type=model_type.format('per-ROI'), 
            PROJECT_PATH=PROJECT_PATH,
            method='mean',
            plot=False,
            voxel_wise=False,
            factor=0,
            fdr=fdr,
            p_val=p_val,
            bootstrap=bootstrap,
            n_times=100, 
            sample_size=50,
        )
