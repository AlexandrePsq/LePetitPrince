import glob
import os
from os.path import join
from .settings import Paths, Extensions, Params

import warnings
warnings.simplefilter(action='ignore')

from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
from scipy import stats
import csv
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from textwrap import wrap

paths = Paths()
extensions = Extensions()
params = Params()


#########################################
############### Utilities ###############
#########################################

def compute(path, overwrite=False):
    """Verify if we can follow computations,
    or not.
    """
    result = True
    if os.path.isfile(path):
        result = overwrite
    return result


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

def log(subject, voxel, alpha, r2):
    """ log stats per fold to a csv file """
    logcsvwriter = csv.writer(open("test.log", "a+"))
    if voxel == 'whole brain':
        logcsvwriter.writerow([subject, voxel, np.mean(r2), np.std(r2),
                            np.min(r2), np.max(r2)])
    else:
        logcsvwriter.writerow([subject, voxel, alpha, r2])



#########################################
############ Data retrieving ############
#########################################

def get_data(language, data_type, subject=None, source='', model=''):
    """Retrieve the requested data depending on its type. This function relies
    on the predefined architecture of the project.
    :language : (str) language used.
    :data_type: (str) Type of data (fMRI, MEG, text, wave) or step of the pipeline 
    (raw-features, features, design-matrices, glm-indiv, ridge-indiv, etc...).
    :subject: (str - optional) Name of the subject (e.g.: sub-xxx).
    :source: (str - optional) Source of acquisition used (fMRI or MEG).
    :model: (str - optional) Name of the model, if the data we want to retrieve.
    is related to a model.
    """
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    output_parent_folder = get_output_parent_folder(data_type, language, source, model)
    if data_type in sub_dir:
        if data_type in ['fMRI', 'MEG']:
            file_pattern = '{2}/func/{0}_{1}_{2}_run*'.format(data_type, language, subject) + extension
        else:
            file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    else:
        file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    data = sorted(glob.glob(join(output_parent_folder, file_pattern)))
    return data


def get_output_parent_folder(data_type, language, source='', model=''):
    """Return the parent folder of the data determined by: its source
    of acquisition, its type, its language and the related model (optional).
    :language : (str) language used.
    :data_type: (str) Type of data (fMRI, MEG, text, wave) or step of the pipeline 
    (raw-features, features, design-matrices, glm-indiv, ridge-indiv, etc...).
    :source: (str - optional) Source of acquisition used (fMRI or MEG).
    :model: (str - optional) Name of the model, if the data we want to retrieve.
    """
    sub_dir = os.listdir(paths.path2data)
    base_path = paths.path2data if data_type in sub_dir else os.path.join(paths.path2derivatives, source)
    return os.path.join(base_path, '{0}/{1}/{2}'.format(data_type, language, model))


def get_path2output(source, data_type, language, model, run_name, extension):
    """Return the path where to save the data depending on: its source
    of acquisition, its type, its language, its extension and the related model and run.
    :language : (str) language used.
    :data_type: (str) Type of data (fMRI, MEG, text, wave) or step of the pipeline 
    (raw-features, features, design-matrices, glm-indiv, ridge-indiv, etc...).
    :source: (str - optional) Source of acquisition used (fMRI or MEG).
    :model: (str - optional) Name of the model, if the data we want to retrieve.
    :run_name: (str) Name of the run (e.g.: run1).
    :extension: (str) Extension of the file saved (.csv, .txt, etc ...).
    """
    output_parent_folder = get_output_parent_folder(data_type, language, source, model)
    check_folder(output_parent_folder)
    return os.path.join(output_parent_folder, '{0}_{1}_{2}_{3}'.format(data_type, language, model, run_name) + extension)

def read_yaml(yaml_path):
    """Read a yaml file.
    :yaml_path: (str) Path to the yaml file.
    """
    with open(yaml_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except :
            print('Error when loading {}...'.format(yaml_path))
            quit()
    return data

def get_design_matrix(models, language):
    """Create the design matrix associated with a set of models.
    :models: (list) List of model names.
    :language: (str) Language that is being used.
    """
    dataframes = []
    features_list = []
    source = 'fMRI'

    for model in models:
        features_list.append(get_data(language, 'features', model=model['model_name'], source=source)) # retrieve the data to transform and append the list of runs (features data) to features_list) 
    runs = list(zip(*features_list)) # list of 9 tuples (1 for each run), each tuple containing the features for all the specified models
    # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]

    # Computing design-matrices
    for i in range(len(runs)):
        merge = pd.concat([pd.read_csv(path2features, header=0)[eval(models[index]['columns2retrieve'])] for index, path2features in enumerate(runs[i])], axis=1) # concatenate horizontaly the read csv files of a run
        dataframes.append(merge)
    return dataframes

def get_category(model_name):
    category = model_name.split('_')[0]
    return category.upper()



#########################################
########## Classical functions ##########
#########################################


def get_scores(model, y_true, x, r2_min=-0.5, r2_max=0.99, pearson_min=-0.2, pearson_max=0.99, output='both'):
    """Return the r2 score and pearson correlation coefficient 
    for each voxel. (=list)
    :model: (sklearn.linear_model) Model trained on a set of voxels.
    :y_true: (np.array) Set of voxels for which to predict the activations.
    :x: (np.array) Design-matrix.
    :r2_min: (float) Min value to filter R2 values.
    :r2_max: (float) Max value to filter R2 values.
    :pearson_min: (float) Min value to filter pearson correlation coefficients.
    :pearson_max: (float) Max value to filter pearson correlation coefficients.
    """
    prediction = model.predict(x)
    r2 = None
    pearson = None
    if output in ['r2', 'both']:
        r2 = r2_score(y_true,
                        prediction,
                        multioutput='raw_values')
        r2 = np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])
    if output in ['pearson', 'both']:
        pearson = [pearsonr(y_true[:,i], prediction[:,i])[0] for i in range(y_true.shape[1])]
        pearson = np.array([0 if (x < pearson_min or x >= pearson_max) else x for x in pearson])
    return r2, pearson


def generate_random_prediction(model, x_test, y_test, n_sample=params.n_sample):
    """Generate predictions over randomly shuffled columns.
    :model: (sklearn.linear_model) Model trained on a set of voxels.
    :x_test: (np.array) Input values for prediction (test).
    :y_test: (np.array) True values to predict.
    :n_sample: (int) Number of sample to generate.
    """
    np.random.seed(1111)
    n_permutations = n_sample
    
    columns_index = np.arange(x_test.shape[1])
    shuffling = []

    # computing permutations
    for _ in range(n_permutations):
        np.random.shuffle(columns_index)
        shuffling.append(columns_index.copy())
    
    # Computing significant values
    distribution_array_r2 = None
    distribution_array_pearson_corr = None
    for index in tqdm(range(n_sample)):
        r2_tmp, pearson_corr_tmp = get_scores(model, y_test, x_test[:, shuffling[index]])
        distribution_array_r2 = r2_tmp if distribution_array_r2 is None else np.vstack([distribution_array_r2, r2_tmp])
        distribution_array_pearson_corr = pearson_corr_tmp if distribution_array_pearson_corr is None else np.vstack([distribution_array_pearson_corr, pearson_corr_tmp])

    return distribution_array_r2, distribution_array_pearson_corr


def get_significativity_value(distribution, distribution_array, alpha):
    """Return significant values for a given distribution by generating prediction 
    over randomly shuffled samples.
    :distribution: (np.array - dim:(1,-1)) Distribution from which to extract significant values.
    :distribution_array: (np.array - dim:(1,-1)) Samples generated to extract significant values from distribution.
    :alpha: (int) Percentile (0-100).
    """
    p_values = (1.0 * np.sum(distribution_array>distribution, axis=0))/distribution_array.shape[0] 
    z_values = np.array([x if np.abs(x) != np.inf else np.sign(x)*10 for x in np.apply_along_axis(lambda x: stats.norm.ppf(1-x, loc=0, scale=1), 0, p_values)])
    mask = (p_values < 1-alpha/100)

    significative = np.zeros(distribution.shape)
    significative[mask] = distribution[mask]
    significative[~mask] = np.nan

    return significative, mask, z_values, p_values


def transform_design_matrices(path):
    """Read a design matrix and perform 
    a desired transformation.
    :path: (str) Path to the matrix.
    """
    dm = pd.read_csv(path, header=0).values
    # Perform the wanted transformation
    #const = np.ones((dm.shape[0], 1))
    #dm = np.hstack((dm, const))
    return dm 

def scale(matrices, scaling_mean=True, scaling_var=True):
    """Standardize a list of matrices.
    :matrices: (list of np.array)
    :scaling_mean: (bool) Center the data.
    :scaling_var: (bool) Standardize the variance.
    """
    for index in range(len(matrices)):
        scaler = StandardScaler(with_mean=scaling_mean, with_std=scaling_var)
        scaler.fit(matrices[index])
        matrices[index] = scaler.transform(matrices[index])
    return matrices


def standardization(matrices, model_name, pca_components=300, scaling=True, pca_type='simple'):
    """Standardize a list of matrices and do a PCA transformation 
    if requested.
    :matrices: (list of np.array)
    :model_name: (str) Name of the model studied.
    :pca_components: (int - optional) Number of components to keep.
    """
    if (matrices[0].shape[1] > pca_components) & (params.pca):
        matrices = scale(matrices)
        print('PCA analysis ({}) running...'.format(pca_type))
        matrices = pca(matrices, model_name, n_components=pca_components) if pca_type=='dual-statis' else simple_pca(matrices, model_name, n_components=pca_components)
        print('PCA done.')
    if scaling:
        matrices = scale(matrices)
    return matrices


def shift(column, n_rows, column_name):
    """Shift by 'n_rows' the elements of a column
    and padd with '0'.
    :column: (pandas.Serie) Pandas serie whom elements need to be shifted.
    :n_rows: (int) Number of indexes to shift by the column elements.
    :column_name: (str) Name of the serie.
    """
    df2 = pd.DataFrame([0]*np.abs(n_rows), columns=[column_name]) 
    tmp = column.iloc[-min(0, n_rows):len(column)-max(0,n_rows)]
    if n_rows >=0:
        result = df2.append(pd.DataFrame(tmp, columns=[column_name]), ignore_index=True)
    else:
        result = pd.DataFrame(tmp, columns=[column_name]).append(df2, ignore_index=True)
    return result



#########################################
################## PCA ##################
#########################################

def simple_pca(X, data_name, n_components=300):
    """Calssical PCA. Done on the concatenated set 
    of matrices given as a list.
    :X: (list of np.array) List of matrices on which to perform the PCA.
    :data_name: (str) Name of the model studied.
    :n_components: (int - optional) Number of components to keep.
    """
    lengths = [m.shape[0] for m in X]
    X_all = np.vstack(X)
    pca = PCA(n_components=n_components)
    pca.fit(X_all)
    # plot explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.savefig(os.path.join(paths.path2derivatives, 'fMRI', 'raw-features', 'explained_variance_{}_{}.png'.format(data_name, n_components)))
    plt.close()
    # extract components
    X_all = pca.transform(X_all)
    index = 0
    result = []
    for i in lengths:
        result.append(X_all[index:index+i,:])
        index += i
    return result


def pca(X, data_name, n_components=300):
    """Compute a Dual-STATIS analysis. It takes account of the 
    similarities between the variance-covariance matrices of the groups.
    See paper:
    General overview of methods of analysis of multi-group datasets
    Aida Eslami, El Mostafa Qannari, Achim Kohler, Stephanie Bougeard
    :X: (list of np.array) List of matrices on which to perform the PCA.
    :data_name: (str) Name of the model studied.
    :n_components: (int - optional) Number of components to keep.
    """
    M = len(X) # number of groups
    # Computing variance-covariance matrix for each group
    cov_matrices = [np.cov(matrix, rowvar=False) for matrix in X]
    R = np.zeros((M, M))
    for i in range(M):
        for k in range(M):
            R[i,k] = np.trace(np.dot(cov_matrices[i], cov_matrices[k]))
    # Computing alphas
    eig_values, eig_vectors = np.linalg.eig(R)
    alphas = eig_vectors[:, np.argmax(eig_values)] # eigen vector associated with the largest eigen value
    # 'Mean' variance-covariance matrix construction
    Vc = np.zeros(cov_matrices[0].shape)
    for index in range(len(cov_matrices)):
        Vc = np.add(Vc, np.dot(alphas[index], cov_matrices[index]))
    # spectral decomposition of Vc
    eig_values_Vc, A = np.linalg.eig(Vc)
    # u,s,v = np.linalg.svd(X_std.T)
    # diag_matrix = np.diag(eig_values_Vc)
    #############################
    # checking the directions of variance
    result = pd.DataFrame(data=[], columns=['index-{}'.format(j) for j in range(len(eig_values_Vc))]+['run']) 
    for index, matrix in enumerate(cov_matrices):
        result.loc[len(result)] = [np.dot(np.dot(A[:,i], matrix), A[:,i]) for i in range(len(eig_values_Vc))] + [index]
    result.to_csv("/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/variance_per_run_{}.csv".format(data_name))
    #############################
    eig_pairs = [(np.abs(eig_values_Vc[i]), A[:,i]) for i in range(len(eig_values_Vc))]
    tot = sum(np.abs(eig_values_Vc))
    var_exp = [(val / tot)*100 for val in sorted(np.abs(eig_values_Vc), reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    ########## check for n_components ##########
    var_model = sum([eig_pairs[index][0] for index in range(n_components)]/tot)*100
    plt.plot(cum_var_exp)
    plt.xlabel('eigenvalue number')
    plt.ylabel('explained variance (%)')
    plt.axhline(y=var_model, color='g', linestyle='--', label='variance explained by the model: {0:.2f}%'.format(var_model))
    plt.axvline(x=n_components, color='r', linestyle='--', label='number of components: {}'.format(n_components))
    plt.title('\n'.join(wrap(data_name)))
    plt.legend()
    plt.savefig(os.path.join(paths.path2derivatives, 'fMRI', data_name+ '_pca_{}.png'.format(n_components)))
    ##################################################
    projected_matrices = []
    projector = eig_pairs[0][1].reshape(-1, 1)
    for index in range(1, n_components):
        projector = np.hstack((projector, eig_pairs[index][1].reshape(-1, 1)))
    for matrix in X:
        projected_matrices.append(np.array(list(map(lambda y: y.real, np.dot(matrix, projector)))))
        
    # normalizing each matrix
    projected_matrices = scale(projected_matrices)
    return projected_matrices
