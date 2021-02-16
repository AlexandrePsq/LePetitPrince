import os
import glob
import itertools
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import nistats
import scipy
import seaborn as sns
import nibabel as nib
import nilearn
from nilearn.image import load_img, mean_img, index_img, threshold_img, math_img, smooth_img, new_img_like
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn import datasets
from scipy.stats import norm
from nilearn.surface import vol_to_surf
from utils import *
import hdbscan
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA 
from sklearn import manifold
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Ridge
from nilearn.masking import compute_epi_mask, apply_mask
from nilearn.signal import clean
from nilearn.image import math_img, mean_img, resample_to_img, index_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_glass_brain, plot_img




language = 'english'
PROJECT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
OUTPUT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/{language}"
INPUT_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/stimuli-representations/{language}"
FMRIDATA_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI/{language}"
MASKER_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/global_masker_95%_{language}"
SMOOTHED_MASKER_PATH = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/smoothed_global_masker_{language}"
path_to_beta_maps_template = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english/{subject_name}/bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers/{subject_name}_bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers_coef_.npy"
mask_template = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english/{subject_name}/bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers/{subject_name}_bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers_R2.nii.gz"
#path_to_beta_maps_template = "/Users/alexpsq/Code/Parietal/maps/{subject_name}_bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers_coef_.npy"
#mask_template = "/Users/alexpsq/Code/Parietal/maps/{subject_name}_bert-base-cased_pre-7_1_post-0_norm-None_norm-inf_temporal-shifting-0_{subject_id}_hidden-all-layers_R2.nii.gz"
#saving_folder = "/Users/alexpsq/Code/Parietal/maps/"
saving_folder =  "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/clustering/"
all_reduction_method = ["umap", "pca", "ica", "isomap"] # "none", "mds", "sp-emb"
all_clustering_method = ["hdbscan", "agg-clus", "max"]

params = {
        'data':None, 
        'reduction':'umap', 
        'clustering':'hdbscan', 
        'mask':None, 
        'min_cluster_size':50, 
        'min_samples':10, 
        'n_neighbors':4, 
        'min_dist':0.0, 
        'n_components':7, 
        'n_clusters':13, 
        'random_state':1111, 
        'cluster_selection_epsilon':0.5, 
        'affinity_reduc' : 'nearest_neighbors',
        'affinity_cluster':'cosine', 
        'linkage':'average',
        'saving_folder': None,
        'plot_name': None
}



def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
            quit()
    return parameters

def load_atlas(name='cort-prob-2mm'):
    atlas = datasets.fetch_atlas_harvard_oxford(name)
    labels = atlas['labels']
    maps = nilearn.image.load_img(atlas['maps'])
    return maps, labels

def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
def write(path, text, end='\n'):
    """Write in the specified text file."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)

def get_subject_name(id):
    """ Get subject name from id.
    Arguments:
        - id: int
    Returns:
        - str
    """
    if id < 10:
        return 'sub-00{}'.format(id)
    elif id < 100:
        return 'sub-0{}'.format(id)
    else:
        return 'sub-{}'.format(id)

def possible_subjects_id(language):
    """ Returns possible subject id list for a given language.
    Arguments:
        - language: str
    Returns:
        result: list (of int)
    """
    if language=='english':
        result = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
                    94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    elif language=='french':
        result = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30] # TO DO
    elif language=='chineese':
        result = [1] # TO DO
    else:
        raise Exception('Language {} not known.'.format(language))
    return result


class Logger(object):
    """ Framework to log encoding analysis progression and results.
    """
    
    def __init__(self, path):
        self.log_path = path
    
    def report_logs(self, logs, level, end):
        """General reporting function.
        Arguments:
            - logs: str
            - level: str
        """
        write(self.log_path, level, end=': ')
        write(self.log_path, logs, end=end)
    
    def error(self, message):
        """Reports ERROR messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='ERROR', end='\n')
        raise Exception(message)
    
    def warning(self, message):
        """Reports WARNING messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='WARNING', end='\n')

    def info(self, message, end=' '):
        """Reports INFO messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='INFO', end=end)
    
    def validate(self):
        """Validate previous message."""
        write(self.log_path, '--> Done', end='\n')
    
    def report_state(self, message):
        """Report state of a computation."""
        write(self.log_path, message, end=' ')

    def figure(self, array):
        """Reports a figure.
        Arguments:
            - array: np.array
        """
        plt.plot(array)
        plt.savefig(os.path.join(os.path.dirname(self.log_path),'explained_variance.png'))
        plt.close()


def load_masker(path, **kwargs):
    params = read_yaml(path + '.yml')
    mask_img = nib.load(path + '.nii.gz')
    masker = NiftiMasker(mask_img)
    masker.set_params(**params)
    if kwargs:
        masker.set_params(**kwargs)
    masker.fit()
    return masker

def save_masker(masker, path):
    params = masker.get_params()
    params = {key: params[key] for key in ['detrend', 'dtype', 'high_pass', 'low_pass', 'mask_strategy', 
                                            'memory_level', 'smoothing_fwhm', 'standardize',
                                            't_r', 'verbose']}
    nib.save(masker.mask_img_, path + '.nii.gz')
    save_yaml(params, path + '.yml')


def get_roi_mask(atlas_maps, index_mask, labels, path=None, global_mask=None):
    """Return the Niftimasker object for a given ROI based on an atlas.
    Optionally resampled based on a global masker.  
    """
    if path is None:
        path = os.path.join(PROJECT_PATH, 'derivatives/fMRI/ROI_masks', labels[index_mask+1])
        check_folder(path)
    if os.path.exists(path + '.nii.gz') and os.path.exists(path + '.yml'):
        masker = load_masker(path)
    else:
        mask = math_img('img > 50', img=index_img(atlas_maps, index_mask))
        if global_mask:
            global_masker = nib.load(global_mask + '.nii.gz')
            mask = resample_to_img(mask, global_masker, interpolation='nearest')
            params = read_yaml(global_mask + '.yml')
            params['detrend'] = False
            params['standardize'] = False
        masker = NiftiMasker(mask)
        if global_mask:
            masker.set_params(**params)
        masker.fit()
        save_masker(masker, path)
    return masker

def compute_global_masker(files, **kwargs): # [[path, path2], [path3, path4]]
    """Returns a NiftiMasker object from list (of list) of files.
    Arguments:
        - files: list (of list of str)
    Returns:
        - masker: NiftiMasker
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.95', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = NiftiMasker(global_mask, **kwargs)
    masker.fit()
    return masker

def fetch_masker(masker_path, language, path_to_fmridata, path_to_input, logger=None, **kwargs):
    """ Fetch or compute if needed a global masker from all subjects of a
    given language.
    Arguments:
        - masker_path: str
        - language: str
        - path_to_input: str
        - path_to_fmridata: str
        - logger: Logger
    """
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        if logger:
            logger.report_state(" loading existing masker...")
        masker = load_masker(masker_path, **kwargs)
    else:
        if logger:
            logger.report_state(" recomputing masker...")
        fmri_runs = {}
        subjects = [get_subject_name(id) for id in possible_subjects_id(language)]
        for subject in subjects:
            _, fmri_paths = fetch_data(path_to_fmridata, path_to_input, subject, language)
            fmri_runs[subject] = fmri_paths
        masker = compute_global_masker(list(fmri_runs.values()), **kwargs)
        save_masker(masker, masker_path)
    return masker

def heat_map(path_to_beta_maps, aggregate=False, title='', saving_path=None):
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps)
    else:
        data = path_to_beta_maps
    plt.figure(figsize=(40,40))
    if aggregate:
        data = aggregate_beta_maps(data, nb_layers=13, layer_size=768)
    sns.heatmap(data)
    plt.title('Heatmap -', title)
    if saving_path is not None:
        plt.savefig(saving_path)
        plt.close()
    else:
        plt.show()

def resample_beta_maps(path_to_beta_maps, original_masker, new_masker):
    """Resample beta map with a new masker.
    path_to_beta_maps: #voxels x #features"""
    if isinstance(original_masker, str):
        original_masker = load_masker(original_masker)
    if isinstance(new_masker, str):
        new_masker = load_masker(new_masker)
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps)
    else:
        data = path_to_beta_maps
    print("Original data has dimension: ", data.shape)
    imgs = original_masker.inverse_transform([data[:, i] for i in range(data.shape[1])])
    new_data = new_masker.transform(imgs) # dimension: #samples x #voxels
    print("New data has dimension: ", new_data.T.shape)
    return new_data.T
    
def create_subject_mask(mask_template, subject_name, subject_id, global_masker, threshold=75):
    mask_img = nib.load(mask_template.format(subject_name=subject_name, subject_id=subject_id))
    mask_tmp = global_masker.transform(mask_img)
    mask = np.zeros(mask_tmp.shape)
    mask[mask_tmp > np.percentile(mask_tmp, threshold)] = 1
    mask = mask.astype(int)
    mask = mask[0].astype(bool)
    return mask

def aggregate_beta_maps(data, nb_layers=13, layer_size=768, aggregation_type='average'):
    """Clustering of voxels or ROI based on their beta maps.
    maps: (#voxels x #features)
    """
    if isinstance(data, str):
        data = np.load(data)
    nb_layers = nb_layers
    layer_size = layer_size
    if aggregation_type=='average':
        result = np.zeros((data.shape[0], nb_layers))
        for index in range(nb_layers):
            result[:, index] = np.mean(data[:, layer_size * index : layer_size * (index + 1)], axis=1)
    elif aggregation_type=='pca':
        result = PCA(n_components=50, random_state=1111).fit_transform(data)
    elif aggregation_type=='umap':
        result = np.hstack([umap.UMAP(n_neighbors=5, min_dist=0., n_components=2, random_state=1111, metric='cosine', output_metric='euclidean').fit_transform(data[:, layer_size * index : layer_size * (index + 1)]) for index in range(nb_layers)])
    return result

def plot_roi_img_surf(surf_img, saving_path, plot_name, mask=None, labels=None, inflated=False, compute_surf=True, colorbar=True, **kwargs):
    fsaverage = datasets.fetch_surf_fsaverage()
    if compute_surf:
        surf_img = vol_to_surf(surf_img, fsaverage[kwargs['surf_mesh']], interpolation='nearest', mask_img=mask)
        surf_img[np.isnan(surf_img)] = -2
    if labels is not None:
        surf_img = np.round(surf_img)
        
    surf_img += 2
    if saving_path:
        plt.close('all')
        plt.hist(surf_img, bins=50)
        plt.title(plot_name)
        plt.savefig(saving_path + "hist_{}.png".format(plot_name))
    plt.close('all')
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
    plotting.show()

def scatter3d(data, cs, colorsMap='jet', reduction_type='', other=None, **kwargs):
    """3D scatter plot with random color to distinguish the shape."""
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', c=scalarMap.to_rgba(cs), **kwargs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(other['elevation'], other['azimuth'])

    #ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.title('Activations embedding for {}'.format(reduction_type), fontsize=24)
    ax.view_init(other['elevation'], other['azimuth'])
    interact()

def plot_reduction(data, plot_type='2D', reduction_type='', other=None, **kwargs):
    """kwargs includes: 
     - s=5
     - c='density'
     - cmap='Spectral'
    """
    plt.close('all')
    if plot_type=='2D':
        plt.scatter(data[:, 0], data[:, 1], **kwargs)
        interact()
    elif plot_type=='3D':
        scatter3d(data, results, colorsMap='jet', reduction_type=reduction_type, other=other, **kwargs)


def plot_reduc(path_to_beta_maps, n_neighbors=30, min_dist=0.0, n_components=3, random_state=1111, cluster_selection_epsilon=0.0):
    """Clustering of voxels or ROI based on their beta maps.
    """
    if isinstance(path_to_beta_maps, str):
        data = np.load(path_to_beta_maps).T
    else:
        data = path_to_beta_maps.T
    
    plot_kwargs = {
        's':2,
        #'c':'density',
        'cmap':'Spectral'
    }
    data = data.T
    print("UMAP...")
    umap_result_3D = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, metric='manhattan').fit_transform(data)
    plot_reduction(umap_result_3D, plot_type='3D', reduction_type='UMAP', other={'elevation': 15, 'azimuth': 15}, **plot_kwargs)
    
def reduce(data, reduction, n_neighbors=30, min_dist=0.0, n_components=10, random_state=1111, affinity_reduc='nearest_neighbors', metric_r='cosine', output_metric='euclidean'):
    """Reduce dimemnsion of beta maps.
    """
    gc.collect()
    if reduction=="umap":
        data_reduced = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, metric=metric_r, output_metric=output_metric).fit_transform(data)
        standard_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state, metric=metric_r, output_metric=output_metric).fit_transform(data)
    elif reduction=="pca":
        data_reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(data)
        standard_embedding = PCA(n_components=2, random_state=random_state).fit_transform(data)
    elif reduction=="ica":
        data_reduced = FastICA(n_components=n_components, random_state=random_state).fit_transform(data)
        standard_embedding = FastICA(n_components=2, random_state=random_state).fit_transform(data)
    elif reduction=="isomap":
        data_reduced = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=-1).fit_transform(data)
        standard_embedding = manifold.Isomap(n_components=2, n_neighbors=n_neighbors, n_jobs=-1).fit_transform(data)
    elif reduction=="mds":
        data_reduced = manifold.MDS(n_components=n_components, random_state=random_state, n_jobs=-1).fit_transform(data)
        standard_embedding = manifold.MDS(n_components=2, random_state=random_state, n_jobs=-1).fit_transform(data)
    elif reduction=="sp-emb":
        data_reduced = manifold.SpectralEmbedding(n_components=n_components, random_state=random_state, n_neighbors=n_neighbors, n_jobs=-1).fit_transform(data)
        standard_embedding = manifold.SpectralEmbedding(n_components=2, random_state=random_state, n_neighbors=n_neighbors, n_jobs=-1, affinity=affinity_reduc).fit_transform(data)
    else:
        data_reduced = data
    return data_reduced, standard_embedding
    
def cluster(data_reduced, mask, clustering, min_cluster_size=20, min_samples=10, n_clusters=13, random_state=1111, cluster_selection_epsilon=0.0, affinity_cluster='cosine', linkage='average', saving_folder=None, plot_name=None, metric_c='euclidean', memory=None):
    """Clustering of voxels or ROI based on their beta maps.
    """    
    gc.collect()
    if clustering=='hdbscan':
        labels_ = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, metric=metric_c).fit_predict(data_reduced)
    elif clustering=='agg-clus':
        #kneighbors_graph_ = kneighbors_graph(data_, n_neighbors=5)
        sc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity_cluster, memory=memory) #  connectivity=kneighbors_graph_
        clustering = sc.fit(data_reduced)
        labels_ = clustering.labels_
    elif clustering=="max":
        labels_ = np.argmax(data_reduced, axis=1)
    all_labels = np.zeros(len(mask))
    all_labels[mask] = labels_
    all_labels[~mask] = -2
    plot_name += '_nb_clusters-{}'.format(len(np.unique(all_labels)))
        
    return all_labels, labels_, plot_name
    
def plot_scatter(labels, standard_embedding, plot_name, saving_folder):
    # Printing scatter plot
    plt.close('all')
    clustered = (labels >= 0)
    colorsMap='jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(labels), vmax=max(labels))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.scatter(standard_embedding[clustered, 0],
                standard_embedding[clustered, 1],
                c=scalarMap.to_rgba(labels[clustered]),
                s=0.1,
                cmap='Spectral')
    plt.title(plot_name)
    check_folder(saving_folder)
    if saving_folder is not None:
        plt.savefig(saving_folder + "scatter_{}.png".format(plot_name))
    plt.show()
    plt.close('all')

def create_name(subject_name, params):
    keys = [
            'reduction',
            'clustering',
            'min_cluster_size',
            'n_neighbors',
            'n_components',
            'affinity_cluster',
            'linkage',
            'min_samples',
            'metric_r',
            'metric_c',
            'output_metric'
    ]
    keys_surname = [
            'red',
            'clust',
            'min-c-size',
            'n-neigh',
            'n-comp',
            'aff-clus',
            'link',
            'min-samples',
            'metric-r',
            'metric-c',
            'out-metric'
    ]
    name = subject_name
    for index, key in enumerate(keys):
        name += '_' + keys_surname[index] + '-' + str(params[key])
    return name

def save_results(data, all_original_length, saving_folder, plot_name, global_masker, mask, **kwargs):
    """Save surface plot."""
    gc.collect()
    if saving_folder is not None:
        check_folder(os.path.join(saving_folder, 'labels'))
        np.save(os.path.join(saving_folder, 'labels', plot_name+'.npy'), data)
    i = 0
    results = []
    nb_clusters = len(np.unique(data))
    print('Input shape: ', data.shape)
    print('Number of clusters: ', nb_clusters)
    for l in all_original_length:
        results.append(get_prob_matrix(data[i: i+l], nb_clusters))
        i += l
    l = results[0].shape[1]
    print('Number of voxels:', l)
    for index, item in enumerate(results):
        assert item.shape[1]==l
    matrix_ = np.stack(results, axis=0)
    prob_matrix = np.mean(matrix_, axis=0)
    argmax_matrix = np.argmax(prob_matrix, axis=0)
    #final_matrix = np.zeros((nb_clusters, len(argmax_matrix)))
    #for index, i in enumerate(rois):
    #    final_matrix[i, index] = 1
    print('Effective number of clusters:', len(np.unique(argmax_matrix)))
    
    img = global_masker.inverse_transform(argmax_matrix)
    # Cleaning memory
    gc.collect()
    del matrix_
    del prob_matrix
    del argmax_matrix
    del results
    plot_roi_img_surf(img, saving_folder, plot_name, mask=mask, labels='round', inflated=False, compute_surf=True, colorbar=True, **kwargs)
    #plot_roi_img_surf(img, saving_folder, plot_name, mask=None, labels=np.unique(results), inflated=False, compute_surf=True, colorbar=True, **kwargs)

def get_connectivity_matrix(labels):
    if isinstance(labels, str):
        labels = np.load(labels)
    result = np.equal.outer(labels,labels).astype(int)
    return result

def get_prob_matrix(labels, nb_clusters):
    if isinstance(labels, str):
        labels = np.load(labels)
    nb_voxels = len(labels)
    prob_matrix = np.zeros((nb_clusters, nb_voxels))
    for index, i in enumerate(range(nb_clusters)):
        prob_matrix[index, labels==i] = 1
    return prob_matrix

def get_labels(subject_name, reduction, clustering, saving_folder, params):
    """Retrieve computed labels for a given subject."""
    saving_folder = os.path.join(saving_folder, '_'.join([reduction, clustering]) + '/')
    plot_name = create_name(subject_name, params)
    path = sorted(glob.glob(os.path.join(saving_folder, 'labels', plot_name + '*.npy')))
    return path

def get_images(subject_name, reduction, clustering, saving_folder, params):
    """Retrieve computed labels for a given subject."""
    saving_folder = os.path.join(saving_folder, '_'.join([reduction, clustering]) + '/')
    plot_name = create_name(subject_name, params)
    path = sorted(glob.glob(os.path.join(saving_folder, plot_name + '*.png')))
    return path


logger = Logger(os.path.join(PROJECT_PATH, 'logs.txt'))
logger.info("Loading data...")
global_masker_50 = fetch_masker(f"{PROJECT_PATH}/global_masker_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)
global_masker_95 = fetch_masker(f"{PROJECT_PATH}/global_masker_95%_{language}"
, language, FMRIDATA_PATH, INPUT_PATH, smoothing_fwhm=None, logger=logger)

atlas_maps, labels = load_atlas() # load harvard-oxford atlas named'cort-prob-2mm'
x_labels = labels[1:]

subjects = [get_subject_name(sub_id) for sub_id in possible_subjects_id(language)]

params.update({'reduction': 'umap', 
               'clustering': 'hdbscan', 
               'n_clusters': 4, 
               'linkage': 'ward', 
               'affinity_cluster': 'euclidean'})

view = 'left' #left
kwargs = {
    'surf_mesh': f'pial_{view}', # pial_right, infl_left, infl_right
    'surf_mesh_type': f'pial_{view}',
    'hemi': view, # right
    'view':'lateral', # medial
    'bg_map': f'sulc_{view}', # sulc_right
    'bg_on_data':True,
    'darkness':.8
}

original_masker = global_masker_50
new_masker = global_masker_95
original_masker.set_params(detrend=False, standardize=False)
new_masker.set_params(detrend=False, standardize=False)

all_data_umap = []
all_data_pca = []
all_masks = []
all_original_length = []
folder = os.path.join(PROJECT_PATH, 'test')

logger.validate()
logger.info("Computing data and mask for different aggregation technics...")
#for subject_name in tqdm(subjects):
#    subject_id = int(subject_name.split('-')[-1])
#    path_to_beta_maps = path_to_beta_maps_template.format(subject_name=subject_name, subject_id=subject_id)
#    mask = create_subject_mask(mask_template, subject_name, subject_id, global_masker_95, threshold=75)
#    
#    data = np.load(path_to_beta_maps)
#    all_masks.append(mask)
#    all_data_umap.append(resample_beta_maps(aggregate_beta_maps(np.vstack(data) if mask is not None else data.copy(), nb_layers=13, layer_size=768, aggregation_type='umap'), original_masker, new_masker)[mask, :])
#    all_data_pca.append(resample_beta_maps(aggregate_beta_maps(np.vstack(data) if mask is not None else data.copy(), nb_layers=13, layer_size=768, aggregation_type='pca'), original_masker, new_masker)[mask, :])
#    all_original_length.append(data.shape[0] if mask is not None else data.copy().shape[0])

logger.validate()

logger.info("Saving and loading...")
#np.save(os.path.join(folder, 'all_data_pca.npy'), np.array(all_data_pca))
#np.save(os.path.join(folder, 'all_data_umap.npy'), np.array(all_data_umap))

all_data_pca = np.load(os.path.join(folder, 'all_data_pca.npy'))
all_masks = np.load(os.path.join(folder, 'all_masks.npy'))
all_data = np.load(os.path.join(folder, 'all_data.npy'))
all_original_length = np.load(os.path.join(folder, 'all_original_length.npy'))
logger.validate()

average_mask = math_img('img>0.4', img=mean_img(new_masker.inverse_transform(all_masks))) # take the average mask and threshold at 0.5
params.update({'reduction':'umap',
               'clustering':'hdbscan', 
               'n_components':3,
               'n_clusters':8,
               'min_samples': 10,
               'min_cluster_size': 30,
               'cluster_selection_epsilon': 0.5,
               'min_dist': 0.0,
               'n_neighbors': 3,
               'metric_r': 'cosine',
               'metric_c': 'euclidean',
               'output_metric': 'euclidean'
                              })



for index_data, data in enumerate([np.vstack(all_data), np.vstack(all_data_pca)]):
    logger.info("New king of data...")
    params.update({'data': data,
               'mask': np.hstack(all_masks)
              })
    reduction = 'umap'
    metric_r = 'cosine'
    metric_c = 'euclidean'
    output_metric = 'euclidean' #euclidean
    random_state = 1111
    min_dist = 0.
    n_components = 3
    affinity_reduc = 'nearest_neighbors'
    
    for n_neighbors in [3, 4, 5]:
    
        data_reduced, standard_embedding = reduce(
            data, 
            reduction, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_components=n_components, 
            random_state=random_state, 
            affinity_reduc=affinity_reduc, 
            metric_r=metric_r, 
            output_metric=output_metric)
        
        for clustering in ['agg-clus']:
            
            cluster_selection_epsilon = 0.
            additional = 'average' if index_data==0 else 'pca'
            saving_folder_ = os.path.join(saving_folder, 'data-{}_'.format(additional) + '_'.join([reduction, clustering]) + '/')
            logger.info('_'.join([reduction, clustering, output_metric, metric_r, metric_c]), end='\n')
            
            if clustering=='hdbscan':
            
                for min_samples in [1, 2, 3, 4, 5, 10, 15, 20]:

                    for min_cluster_size in [120, 150, 180, 200, 220, 250, 280, 300]:
                        params.update({'reduction':reduction,
                                                'clustering':clustering, 
                                                'n_components':n_components,
                                                'n_clusters':0,
                                                'min_samples': min_samples,
                                                'min_cluster_size': min_cluster_size,
                                                'cluster_selection_epsilon': cluster_selection_epsilon,
                                                'min_dist': min_dist,
                                                'n_neighbors': n_neighbors,
                                                'metric_r': metric_r,
                                                'metric_c': metric_c,
                                                'output_metric': output_metric
                                                                })
                        plot_name = create_name('all-subjects', params)

                        all_labels, labels_, plot_name = cluster(
                                data_reduced, 
                                params['mask'], 
                                clustering, 
                                min_cluster_size=min_cluster_size, 
                                min_samples=min_samples, 
                                n_clusters=0, 
                                random_state=random_state, 
                                cluster_selection_epsilon=cluster_selection_epsilon, 
                                affinity_cluster='', 
                                linkage='', 
                                saving_folder=saving_folder_, 
                                plot_name=plot_name, 
                                metric_c=metric_c)

                        plot_scatter(labels_, standard_embedding, plot_name, saving_folder_)
                        save_results(all_labels, all_original_length, saving_folder_, plot_name, new_masker, average_mask, **kwargs) #params['saving_folder']
                    
            elif clustering=='agg-clus':
                
                for linkage, affinity_cluster in zip(['ward', 'complete'], ['euclidean', 'cosine']):
                    
                    for n_clusters in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                        
                        params.update({'reduction':reduction,
                                                'clustering':clustering, 
                                                'n_components':n_components,
                                                'n_clusters':n_clusters,
                                                'min_samples': 0,
                                                'min_cluster_size': 0,
                                                'cluster_selection_epsilon': cluster_selection_epsilon,
                                                'min_dist': min_dist,
                                                'n_neighbors': n_neighbors,
                                                'metric_r': metric_r,
                                                'metric_c': metric_c,
                                                'output_metric': output_metric
                                                                })
                        plot_name = create_name('all-subjects', params)

                        all_labels, labels_, plot_name = cluster(
                            data_reduced, 
                            params['mask'], 
                            clustering, 
                            min_cluster_size=0, 
                            min_samples=0, 
                            n_clusters=n_clusters, 
                            random_state=random_state, 
                            cluster_selection_epsilon=cluster_selection_epsilon, 
                            affinity_cluster=affinity_cluster, 
                            linkage=linkage, 
                            saving_folder=saving_folder_, 
                            plot_name=plot_name, 
                            metric_c=metric_c)

                        plot_scatter(labels_, standard_embedding, plot_name, saving_folder_)
                        gc.collect()
                        save_results(all_labels, all_original_length, saving_folder_, plot_name, new_masker, average_mask, **kwargs) #params['saving_folder']




