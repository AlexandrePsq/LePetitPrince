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



    ##################
    ### Parameters ###
    ##################

    y = np.load(os.path.join(fmri_path,'y_run1.npy'))
    x = np.load(os.path.join(design_matrices_path,'x_run1.npy'))

    nb_runs = str(len(fmri_runs))
    nb_voxels = str(y[0].shape[1])
    nb_features = str(x[0].shape[1])
    nb_permutations = str(3000)
    alpha_list = [round(tmp, 5) for tmp in np.logspace(-3, 3, 100)]
    alphas = ','.join([str(alpha) for alpha in alpha_list]) 
    alpha_percentile = str(99)
    features = []
    # features = sorted([['rms_model', 1], ['wordrate_model', 1]]) # add here basic features name + nb of column feature for each!!!
    # could be: features = sorted([['lstm...hidden_first-layer', 300], ['lstm...cell_first-layer', 300]])


    ################
    ### Pipeline ###
    ################

    ### Create the workflow:
    dependencies = []
    jobs = []


    # Merging the results and compute significant r2
    job_merge = Job(command=["python", "merge_results.py", 
                                "--input_folder", derivatives_path, 
                                "--parameters", parameters_path, 
                                "--yaml_files", yaml_files_path,
                                "--nb_runs", nb_runs, 
                                "--nb_voxels", nb_voxels,
                                "--n_permutations", nb_permutations, 
                                "--alpha_percentile", alpha_percentile], 
            name="Merging all the r2 and distribution respectively together.",
            working_directory=scripts_path)


    
    jobs.append(job_merge)

    # Plotting the maps
    job_final = Job(command=["python", "create_maps.py", 
                                "--input", derivatives_path, 
                                "--parameters", parameters_path,
                                "--subject", args.subject,
                                "--fmri_data", fmri_path], 
                        name="Creating the maps.",
                        working_directory=scripts_path)
    jobs.append(job_final)
    dependencies.append((job_merge, job_final))

    workflow = Workflow(jobs=jobs,
                    dependencies=dependencies,
                    root_group=[job_0, cv_alphas, significativity, job_merge, job_final])
                

    Helper.serialize(os.path.join(inputs_path, 'delete.somawf'), workflow)


    ### Submit the workflow to computing resource (configured in the client-server mode)

    controller = WorkflowController("DSV_cluster_{}".format(login), login, password) #"DSV_cluster_ap259944", login, password

    workflow_id = controller.submit_workflow(workflow=workflow,
                                            name="Ridge - LPP")

    
    print("Finished !!!")