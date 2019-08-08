# -*- coding: utf-8 -*-
########################
# python cluster.py --login ap259944 --password XxxxX --fmri_data /home/ap259944/inputs/fmri_data/ --design_matrices /home/ap259944/inputs/design_matrices/ --model_name lstm_wikikristina_embedding-size_600_nhid_600_nlayers_1_dropout_02_hidden_first-layer --subject sub-057
########################

from soma_workflow.client import Job, Workflow, Helper, Group, WorkflowController, FileTransfer, SharedResourcePath
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
import nibabel as nib
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse



############################
### Functions definition ###
############################

def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0).values
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 


def standardization(matrices, model_name, pca_components="300"):
    if (pca_components) & (matrices[0].shape[1] > int(pca_components)):
        print('PCA analysis running...')
        matrices = pca(matrices, model_name, n_components=int(pca_components))
        print('PCA done.')
    else:
        print('Skipping PCA.')
        for index in range(len(matrices)):
            scaler = StandardScaler(with_mean=True, with_std=False)
            scaler.fit(matrices[index])
            matrices[index] = scaler.transform(matrices[index])
    return matrices


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
    #############################
    eig_pairs = [(np.abs(eig_values_Vc[i]), A[:,i]) for i in range(len(eig_values_Vc))]
    eig_pairs.sort()
    eig_pairs.reverse()
    tot = sum(np.abs(eig_values_Vc))
    ##################################################
    projected_matrices = []
    projector = eig_pairs[0][1].reshape(-1, 1)
    for index in range(1, n_components):
        projector = np.hstack((projector, eig_pairs[index][1].reshape(-1, 1)))
    for matrix in X:
        projected_matrices.append(np.dot(matrix, projector))
    # normalizing each matrix
    for index in range(len(projected_matrices)):
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaler.fit(projected_matrices[index])
        projected_matrices[index] = scaler.transform(projected_matrices[index])
    return projected_matrices
            

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker


def create_maps(masker, distribution, distribution_name, subject, output_parent_folder, vmax=0.2, pca='', voxel_wise=False):
    model = os.path.basename(output_parent_folder)
    language = os.path.basename(os.path.dirname(output_parent_folder))
    data_type = os.path.basename(os.path.dirname(os.path.dirname(output_parent_folder)))

    img = masker.inverse_transform(distribution)

    pca = 'pca_' + str(pca) if (pca!='') else 'no_pca'
    voxel_wise = 'voxel_wise' if voxel_wise else 'not_voxel_wise'
    
    path2output_raw = os.path.join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.nii.gz')
    path2output_png = os.path.join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(data_type, language, model, distribution_name, pca, voxel_wise, subject)+'.png')

    nib.save(img, path2output_raw)

    display = plot_glass_brain(img, display_mode='lzry', threshold=0, colorbar=True, black_bg=True, vmin=0., vmax=vmax)
    display.savefig(path2output_png)
    display.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nUse cluster nodes to perform the Ridge analysis.""")
    parser.add_argument("--login", type=str, default=None, help="Login to connect to the cluster.")
    parser.add_argument("--password", type=str, default=None, help="Password to connect to the cluster.")
    parser.add_argument("--fmri_data", type=str, default='', help="Path to fMRI data directory.")
    parser.add_argument("--design_matrices", type=str, default='', help="Path to design-matrices directory(for a given model).")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model.")
    parser.add_argument("--subject", type=str, default='sub-057', help="Subject name.")
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

    inputs_path = "/home/ap259944/inputs/"
    scripts_path = "/home/ap259944/inputs/scripts/"
    fmri_path = "/home/ap259944/inputs/y/"
    design_matrices_path = "/home/ap259944/inputs/x/"
    derivatives_path = "/home/ap259944/derivatives/"
    shuffling_path = "/home/ap259944/derivatives/shuffling.npy"
    r2_path = "/home/ap259944/derivatives/r2/"
    distribution_path = "/home/ap259944/derivatives/distribution/"
    yaml_files_path = "/home/ap259944/derivatives/yaml_files/"
    output_path = "/home/ap259944/outputs/"

    design_matrices = sorted(glob.glob(os.path.join(args.design_matrices, '*run*')))
    fmri_runs = sorted(glob.glob(os.path.join(args.fmri_data, '*run*')))


    ###########################
    ### Data transformation ###
    ###########################

    masker = compute_global_masker(list(fmri_runs))
    matrices = [transform_design_matrices(run) for run in design_matrices] # list of design matrices (dataframes) where we added a constant column equal to 1
    x = standardization(matrices, args.model_name, pca_components=args.pca)
    y = [masker.transform(f) for f in fmri_runs] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column

    for index in range(len(y)):
        np.save(os.path.join(fmri_path, 'y_run{}.npy'.format(index)), y[index])
        np.save(os.path.join(design_matrices_path, 'x_run{}.npy'.format(index)), y[index])

    ##################
    ### Parameters ###
    ##################

    nb_runs = str(len(fmri_runs))
    nb_voxels = str(y[0].shape[1])
    nb_features = str(x[0].shape[1])
    nb_permutations = str(300)
    alpha_list = np.logspace(-3, 3, 100)
    alphas = ','.join([str(alpha) for alpha in alpha_list]) 
    alpha_percentile = str(95)


    ####################
    ### Sanity check ###
    ####################

    all_paths = [inputs_path, 
                    scripts_path,
                    fmri_path, 
                    design_matrices_path, 
                    derivatives_path, 
                    shuffling_path, 
                    r2_path, 
                    distribution_path, 
                    yaml_files_path, 
                    output_path]
    for path in all_paths:
        check_folder(path)


    ################
    ### Pipeline ###
    ################

    ### FileTransfers
    # FileTransfer creation for input/output files
    scripts_directory = FileTransfer(is_input=True,
                        client_path=scripts_path,
                        name="working directory")
    fmri_directory = FileTransfer(is_input=True,
                        client_path=fmri_path,
                        name="fmri data directory")
    design_matrices_directory = FileTransfer(is_input=True,
                        client_path=design_matrices_path,
                        name="design-matrices directory")
    derivatives_directory = FileTransfer(is_input=True,
                        client_path=derivatives_path,
                        name="derivatives directory")
    shuffling_path = FileTransfer(is_input=True,
                        client_path=shuffling_path,
                        name="shuffling file")
    r2_directory = FileTransfer(is_input=True,
                        client_path=r2_path,
                        name="R2 directory")
    distribution_directory = FileTransfer(is_input=True,
                        client_path=distribution_path,
                        name="distribution directory")
    yaml_files_directory = FileTransfer(is_input=True,
                        client_path=yaml_files_path,
                        name="yaml files directory")
    output_directory = FileTransfer(is_input=False,
                        client_path=output_path,
                        name="Outputs directory")


    ### Create the workflow:
    dependencies = []
    jobs = []

    # first job: split the dataset
    job_0 = Job(command=["python", "shuffling_preparation.py", 
                            "--nb_features", nb_features,
                            "--shuffling", shuffling_path,
                            "--n_permutations", nb_permutations], 
                name="Preparing permutations for significance analysis", 
                referenced_input_files=[scripts_directory],
                referenced_output_files=[shuffling_path], 
                working_directory=scripts_directory)

    jobs.append(job_0)

    # significativity retrieval 
    job_generator = Job(command=["python", "generate_jobs.py", 
                                    "--x", design_matrices_directory, 
                                    "--y", fmri_directory, 
                                    "--alphas", alphas, 
                                    "--output", output_directory, 
                                    "--shuffling", shuffling_path, 
                                    "--nb_permutations", nb_permutations, 
                                    "--alpha_percentile", alpha_percentile, 
                                    "--login", login, 
                                    "--password", password], 
                        name="Launch the internal workflow to compute R2 and voxels distribution", 
                        referenced_input_files=[scripts_directory],
                        referenced_output_files=[r2_directory, distribution_directory], 
                        working_directory=scripts_directory)


    # cross-validation on alpha for each split 
    group_cv_alphas = []

    for run in range(int(nb_runs)):
        indexes = np.arange(int(nb_runs))
        indexes = ','.join([str(i) for i in np.delete(indexes, run, 0)]) 
        job = Job(command=["python", "cv_alphas.py", 
                                "--indexes", indexes, 
                                "--x", design_matrices_directory, 
                                "--y", fmri_directory, 
                                "--run", str(run), 
                                "--alphas", alphas, 
                                "--output", yaml_files_directory],  
                name="Alphas CV - split {}".format(run + 1), 
                referenced_input_files=[scripts_directory, design_matrices_directory, fmri_directory],
                referenced_output_files=[yaml_files_directory], 
                working_directory=scripts_directory)

        group_cv_alphas.append(job)
        jobs.append(job)
        dependencies.append((job_0, job))
        dependencies.append((job, job_generator))

    jobs.append(job_generator)
                
    # Merging the results and compute significant r2
    job_final = Job(command=["python", "merge_results.py", 
                                "--input_r2", r2_directory, 
                                "--input_distribution", distribution_directory, 
                                "--output", output_directory, 
                                "--nb_runs", nb_runs, 
                                "--n_permutations", nb_permutations, 
                                "--alpha_percentile", alpha_percentile], 
            name="Merging all the r2 and distribution respectively together.", 
            referenced_input_files=[scripts_directory, r2_directory, distribution_directory],
            referenced_output_files=[output_directory], 
            working_directory=scripts_directory)

    jobs.append(job_final)
    dependencies.append((job_generator, job_final))

    cv_alphas = Group(elements=group_cv_alphas,
                        name="CV on alphas")

    workflow = Workflow(jobs=jobs,
                    dependencies= dependencies,
                    root_group=[job_0, cv_alphas, job_generator, job_final])


    ### Submit the workflow to computing resource (configured in the client-server mode)

    controller = WorkflowController("DSV_cluster_ap259944", login, password)

    workflow_id = controller.submit_workflow(workflow=workflow,
                                            name="Ridge - LPP")

    # You may use the gui or manually transfer the files:
    manual = True
    if manual:
        Helper.transfer_input_files(workflow_id, controller)
        Helper.wait_workflow(workflow_id, controller)
        Helper.transfer_output_files(workflow_id, controller)
        Helper.delete_all_workflows(controller)

    print("Finished !!!")


    #####################
    ### Plotting maps ###
    #####################
    output_parent_folder = os.path.join(output_path, 'maps')
    alphas = np.mean(np.load(os.path.join(output_path, 'voxel2alpha.npy')), axis=0)
    r2 = np.load(os.path.join(output_path, 'r2.npy'))
    z_values = np.load(os.path.join(output_path, 'z_values.npy'))
    significant_r2 = np.load(os.path.join(output_path, 'significant_r2.npy'))
    pca = int(args.pca)
    
    # creating maps
    create_maps(masker, alphas, 'alphas', args.subject, output_parent_folder, pca=pca) # alphas # argument deleted: , vmax=5e3
    create_maps(masker, r2, 'r2', args.subject, output_parent_folder, vmax=0.2, pca=pca,  voxel_wise=True) # r2 test
    create_maps(masker, z_values, 'z_values', args.subject, output_parent_folder, pca=pca, voxel_wise=True) # p_values
    create_maps(masker, significant_r2, 'significant_r2', args.subject, output_parent_folder, vmax=0.2, pca=pca, voxel_wise=True) # r2_significative


    