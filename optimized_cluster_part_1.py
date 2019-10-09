# -*- coding: utf-8 -*-
########################
# python cluster.py --login ap259944 --password XxxxX --fmri_data /home/ap259944/inputs/fmri_data/ --design_matrices /home/ap259944/inputs/design_matrices/ --model_name lstm_wikikristina_embedding-size_600_nhid_600_nlayers_1_dropout_02_hidden_first-layer --subject sub-057
########################

from soma_workflow.client import Job, Workflow, Helper, Group, WorkflowController
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nilearn.masking import compute_epi_mask # cause warning
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from sklearn.model_selection import LeaveOneOut
from nilearn.plotting import plot_glass_brain
import nibabel as nib
import yaml
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse



############################
### Functions definition ###
############################

def check_folder(path):
    # Create adequate folders if necessary
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0).values
    # add the constant
    # const = np.ones((dm.shape[0], 1))
    # dm = np.hstack((dm, const))
    return dm 


def standardization(matrices, model_name):
    for index in range(len(matrices)):
        scaler = StandardScaler(with_mean=True, with_std=True) # with_std=True => computation slower
        scaler.fit(matrices[index])
        matrices[index] = scaler.transform(matrices[index])
    return matrices


def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nUse cluster nodes to perform the Ridge analysis.""")
    parser.add_argument("--login", type=str, default=None, help="Login to connect to the cluster.")
    parser.add_argument("--password", type=str, default=None, help="Password to connect to the cluster.")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model.")
    parser.add_argument("--subject", type=str, default='sub-057', help="Subject name.")
    parser.add_argument("--split_features", action='store_true', default=False, help="Full analysis of all the features included in the model.")
    parser.add_argument("--pca", type=str, default=None, help="Number of components to keep for the PCA.")

    args = parser.parse_args()


    ###################
    ### Credentials ###
    ###################

    login = args.login 
    password = args.password 


    ######################
    ### Data retrieval ###
    ######################

    inputs_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
    scripts_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/code/utilities"
    fmri_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI/english/{}/func/".format(args.subject) 
    design_matrices_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/design-matrices/english/{}/".format(args.model_name)
    derivatives_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/".format(args.subject, args.model_name)
    subject_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/".format(args.subject)
    shuffling_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/shuffling.npy".format(args.subject, args.model_name)
    r2_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/r2/".format(args.subject, args.model_name)
    pearson_corr_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/pearson_corr/".format(args.subject, args.model_name)
    distribution_r2_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/distribution_r2/".format(args.subject, args.model_name)
    distribution_pearson_corr_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/distribution_pearson_corr/".format(args.subject, args.model_name)
    yaml_files_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/yaml_files/".format(args.subject, args.model_name)
    output_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/english/{}/{}/outputs/".format(args.subject, args.model_name)
    log_error_path = "/home/{}/soma-workflow/logs/error_log.txt".format(login)
    log_output_path = "/home/{}/soma-workflow/logs/output_log.txt".format(login)

    design_matrices = sorted(glob.glob(os.path.join(design_matrices_path, 'design-matrices_*run*')))
    fmri_runs = sorted(glob.glob(os.path.join(fmri_path, 'fMRI_*run*')))


    ####################
    ### Sanity check ###
    ####################

    all_paths = [inputs_path, 
                    scripts_path,
                    fmri_path, 
                    design_matrices_path, 
                    derivatives_path, 
                    r2_path, 
                    pearson_corr_path,
                    distribution_r2_path, 
                    distribution_pearson_corr_path, 
                    yaml_files_path, 
                    output_path,
                    subject_path,
                    os.path.dirname(log_error_path),
                    os.path.dirname(log_output_path)]
    for path in all_paths:
        check_folder(path)



    ###########################
    ### Data transformation ###
    ###########################

    masker = compute_global_masker(list(fmri_runs))
    matrices = [transform_design_matrices(run) for run in design_matrices] # list of design matrices (dataframes) where we added a constant column equal to 1
    x = standardization(matrices, args.model_name)
    y = [masker.transform(f) for f in fmri_runs] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column

    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    for run in range(len(y)):
        new = np.random.random(y[run].shape[0])/1000
        zero = np.zeros(y[run].shape[0])
        y[run] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, y[run])

    for index in range(len(y)):
        np.save(os.path.join(fmri_path, 'y_run{}.npy'.format(index+1)), y[index])
        np.save(os.path.join(design_matrices_path, 'x_run{}.npy'.format(index+1)), x[index])

    ##################
    ### Parameters ###
    ##################

    nb_runs = str(len(fmri_runs))
    nb_voxels = str(y[0].shape[1])
    nb_features = str(x[0].shape[1])
    nb_permutations = str(100) # should be a multiple of 100
    alpha_list = [round(tmp, 5) for tmp in np.logspace(2, 5, 25)]
    alphas = ','.join([str(alpha) for alpha in alpha_list]) 
    alpha_percentile = str(99)
    # features = []
    # features = [['bert_bucket_layer-a-1', 768], ['bert_bucket_layer-a-2', 768], ['bert_bucket_layer-a-3', 768], ['bert_bucket_layer-a-4', 768], ['bert_bucket_layer-a-5', 768], ['bert_bucket_layer-a-6', 768], ['bert_bucket_layer-a-7', 768], ['bert_bucket_layer-a-8', 768], ['bert_bucket_layer-a-9', 768], ['bert_bucket_layer-b-10', 768], ['bert_bucket_layer-b-11', 768], ['bert_bucket_layer-b-12', 768]]
    features = [['mfcc_model', 39], ['rms_model', 1], ['sentence_onset', 1], ['bottomup_model', 1], ['topdown_model', 1], ['content_words', 1], ['function_words', 1], ['log_frequencies', 1], ['position_in_sentence', 1]] # add here basic features name + nb of column feature for each!!!
    # could be: features = sorted([['lstm...hidden_first-layer', 300], ['lstm...cell_first-layer', 300]])

    parameters_path = os.path.join(derivatives_path, 'parameters.yml')
    parameters = {'models': [], 
                    'nb_runs':nb_runs, 
                    'nb_voxels':nb_voxels, 
                    'nb_features':nb_features, 
                    'nb_permutations':nb_permutations, 
                    'alphas':alphas,
                    'alpha_percentile': alpha_percentile}
    if args.split_features:
        i = 0
        for model in features:
            parameters['models'].append({'name':model[0], 'indexes':[i, i+model[1]]})
            i += model[1]
    parameters['models'].append({'name':'', 'indexes':[0, int(nb_features)]})

    with open(parameters_path, 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)


    ################
    ### Pipeline ###
    ################

    ### Create the workflow:
    dependencies = []
    jobs = []

    # first job: split the dataset
    job_0 = Job(command=["python", "shuffling_preparation.py", 
                            "--nb_features", nb_features,
                            "--output", shuffling_path,
                            "--n_permutations", nb_permutations], 
                name="Preparing permutations for significance analysis",
                working_directory=scripts_path)

    jobs.append(job_0)

    # cross-validation on alpha for each split 
    group_cv_alphas = []
    group_cv_merge = []
    jobs_tmp = []

    for run in range(1, 1+int(nb_runs)):
        indexes = np.arange(1, 1+int(nb_runs))
        logo = LeaveOneOut() # leave on run out !
        cv_index = 1
        indexes_tmp = np.delete(indexes, run-1, 0)
        job_merge_cv = Job(command=["python", "merge_CV.py", 
                            "--indexes", ','.join([str(i) for i in indexes_tmp]), 
                            "--nb_runs", str(len(indexes_tmp)), 
                            "--run", str(run), 
                            "--alphas", alphas, 
                            "--nb_voxels", nb_voxels,
                            "--input", yaml_files_path,
                            "--output", yaml_files_path],  
                    name="Merging alphas CV outputs - split {}".format(run), 
                    working_directory=scripts_path,
                    native_specification="-q Nspin_bigM")
        for train, valid in logo.split(indexes_tmp):
            job = Job(command=["python", "CV_alphas_distributed.py", 
                                "--indexes", ','.join([str(i) for i in indexes_tmp]), 
                                "--train", ','.join([str(i) for i in train]), 
                                "--valid", ','.join([str(i) for i in valid]), 
                                "--x", design_matrices_path, 
                                "--y", fmri_path,
                                "--run", str(run),
                                "--alphas", alphas,
                                "--output", yaml_files_path, 
                                "--nb_voxels", nb_voxels,
                                "--cv_index", str(cv_index)],  
                        name="Alphas CV - split {} - {}".format(run, cv_index), 
                        working_directory=scripts_path,
                        native_specification="-q Nspin_bigM")
            cv_index += 1
            group_cv_alphas.append(job)
            jobs.append(job)
            dependencies.append((job_0, job))
            dependencies.append((job, job_merge_cv))
        group_cv_merge.append(job_merge_cv)
        jobs_tmp.append(job_merge_cv)
    
    jobs += jobs_tmp

    cv_alphas = Group(elements=group_cv_alphas,
                        name="CV on alphas")
    
    cv_merge = Group(elements=group_cv_merge,
                        name="merge CV results")

    workflow = Workflow(jobs=jobs,
                    dependencies=dependencies,
                    root_group=[job_0, cv_alphas, cv_merge])
                

    Helper.serialize(os.path.join(inputs_path, 'optimized_cluster_part_1.somawf'), workflow)


    ### Submit the workflow to computing resource (configured in the client-server mode)

    controller = WorkflowController("DSV_cluster_{}".format(login), login, password) #"DSV_cluster_ap259944", login, password

    workflow_id = controller.submit_workflow(workflow=workflow,
                                            name="Cluster optimized part 1")

    
    print("Finished !!!")


