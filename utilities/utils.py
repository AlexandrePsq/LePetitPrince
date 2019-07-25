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
import csv
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from textwrap import wrap

paths = Paths()
extensions = Extensions()
params = Params()


#########################################
############ Data retrieving ############
#########################################

def get_data(language, data_type, subject=None, source='', model=''):
    # General function for data retrieving
    # Output: list of path to the different data files
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    if data_type in sub_dir:
        base_path = paths.path2data
        if data_type in ['fMRI', 'MEG']:
            file_pattern = '{2}/func/{0}_{1}_{2}_run*'.format(data_type, language, subject) + extension
        else:
            file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    else:
        base_path = join(paths.path2derivatives, source)
        file_pattern = '{}_{}_{}_run*'.format(data_type, language, model) + extension
    data = sorted(glob.glob(join(base_path, '{0}/{1}/{2}'.format(data_type, language, model), file_pattern)))
    return data


def get_output_parent_folder(source, output_data_type, language, model):
    return join(paths.path2derivatives, '{0}/{1}/{2}/{3}'.format(source, output_data_type, language, model))


def get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension):
    return join(output_parent_folder, '{0}_{1}_{2}_{3}'.format(output_data_type, language, model, run_name) + extension)



#########################################
###### Computation functionalities ######
#########################################

def compute(path, overwrite=False):
    # Tell us if we can compute or not
    result = True
    if os.path.isfile(path):
        result = overwrite
    return result


def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        check_folder(os.path.dirname(path))
        os.mkdir(path)



#########################################
################## Log ##################
#########################################

def log(subject, voxel, alpha, r2):
    """ log stats per fold to a csv file """
    logcsvwriter = csv.writer(open("test.log", "a+"))
    if voxel == 'whole brain':
        logcsvwriter.writerow([subject, voxel, np.mean(r2), np.std(r2),
                            np.min(r2), np.max(r2)])
    else:
        logcsvwriter.writerow([subject, voxel, alpha, r2])


#########################################
########## Classical functions ##########
#########################################


def get_r2_score(model, y_true, x, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    r2 = r2_score(y_true,
                    model.predict(x),
                    multioutput='raw_values')
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    # return np.array([0 if (x < r2_min or x >= r2_max) else x for x in r2])
    return r2


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0).values
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 


def shift(column, n_rows, column_name):
    # shift the rows of a column and padd with 0
    df2 = pd.DataFrame([0]*np.abs(n_rows), columns=[column_name]) 
    tmp = column.iloc[-min(0, n_rows):len(column)-max(0,n_rows)]
    if n_rows >=0:
        result = df2.append(pd.DataFrame(tmp, columns=[column_name]), ignore_index=True)
    else:
        print(pd.DataFrame(tmp, columns=[column_name]))
        print(df2)
        result = pd.DataFrame(tmp, columns=[column_name]).append(df2, ignore_index=True)
    return result



#########################################
################## PCA ##################
#########################################
# Compute a Dual-STATIS analysis 
# takes account of the similarities between the variance-covariance matrices of the groups


def pca(X, data_name, n_components=50):
    """
    See paper:
    General overview of methods of analysis of multi-group datasets
    Aida Eslami, El Mostafa Qannari, Achim Kohler, Stephanie Bougeard
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
    ########## testing ##########
    #for matrix in cov_matrices:
    #    for index in range(A.shape[0]):
    #        print(np.dot(np.dot(A[index], matrix), A[index].T))
    #############################
    eig_pairs = [(np.abs(eig_values_Vc[i]), A[:,i]) for i in range(len(eig_values_Vc))]
    eig_pairs.sort()
    eig_pairs.reverse()
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
        projected_matrices.append(np.dot(matrix, projector))
    # normalizing each matrix
    for index in range(len(projected_matrices)):
        scaler = StandardScaler(with_mean=params.scaling_mean, with_std=params.scaling_var)
        scaler.fit(projected_matrices[index])
        projected_matrices[index] = scaler.transform(projected_matrices[index])
    return projected_matrices



##########################################
########## Significativity test ##########
##########################################

def sample_r2(model, x_test, y_test, shuffling, n_sample, alpha_percentile, test=False):
    # receive a trained model, x_test and y_test (test set of the cross-validation).
    # It returns three values (or three list depending on the parameter voxel_wised):
    # r2 value computed on test set, the thresholded value at percentile(alpha_percentile) and 
    # percentile(alpha_percentile)
    if test:
        r2_test = get_r2_score(model, y_test, x_test)
        return r2_test, None, None, None
    else:
        r2_test = get_r2_score(model, y_test, x_test)
        distribution_array = None
        for index in tqdm(range(n_sample)):
            r2_tmp = get_r2_score(model, y_test, x_test.T[shuffling[index]].T)
            distribution_array = r2_tmp if distribution_array is None else np.vstack([distribution_array, r2_tmp])
        return r2_test, distribution_array
        # thresholds = np.percentile(distribution_array, alpha_percentile, axis=0) # list: 1 value for each voxel
        # r2_significative = r2_test.copy()
        # r2_significative[r2_test < thresholds] = 0.
        # return r2_test, r2_significative, thresholds, distribution_array


def get_significativity_value(r2_test_array, distribution_array, alpha_percentile, test=False):
    # receive r2 computed on the test set and r2 significative computed on each set
    # (for each voxel each) and the entire distribution of r2 values computed accross
    # all runs (with shuffled columns) and returns significative r2 computed with the 
    # method that you want.
    r2_final = []
    r2_significative_final = []
    thresholds_final = []

    r2_test = np.mean(r2_test_array, axis=0)
    r2_final.append(r2_test)
    distribution_array = np.mean(distribution_array, axis=0)

    if test:
        r2_significative_final.append(None)
        thresholds_final.append(None)
    else:
        thresholds = np.percentile(distribution_array, alpha_percentile, axis=0) # list: 1 value for each voxel
        r2_significative = r2_test.copy()
        r2_significative[r2_test < thresholds] = 0.
        r2_final.append(r2_test)
        r2_significative_final.append(r2_significative)
        thresholds_final.append(thresholds)

        # alex 2.0
        # Ns = np.count_nonzero(r2_significative_array, axis=0)
        # N = r2_significative_array.shape[0]
        # normalization = (Ns**2 + (N-Ns)**2)/N
        # r2_test = (np.sum(np.multiply(r2_test_array, (Ns * (r2_significative_array > 0) + (N - Ns) * (r2_significative_array == 0))) / N, axis=0))/normalization # list: 1 value for each voxel
        # thresholds = np.percentile(distribution_array, alpha_percentile, axis=0) # list: 1 value for each voxel
        # r2_significative = r2_test.copy()
        # r2_significative[r2_test < thresholds] = 0.
        # r2_final.append(r2_test)
        # r2_significative_final.append(r2_significative)
        # thresholds_final.append(thresholds)
    
    return list(zip(r2_final, r2_significative_final, thresholds_final)), ['threshold the averaged values', 'average the thresholded values', 'weighted average']