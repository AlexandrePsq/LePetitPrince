# -*- coding: utf-8 -*-

import os
import glob
import yaml
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from soma_workflow.client import Job, Workflow, Helper, Group, WorkflowController, FileTransfer, SharedResourcePath



scripts_path = '/Users/alexpsq/Code/inputs/scripts/' # "/home/ap259944/inputs/scripts/"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--x", type=str, default='', help="Path to x folder.")
    parser.add_argument("--y", type=str, default='', help="Path to y folder.")
    parser.add_argument("--alphas", type=str, default=None, help="Alphas values to test during CV (delimiter=',').")
    parser.add_argument("--output", type=str, default=None, help="Output folder.")
    parser.add_argument("--shuffling", type=str, default=None, help="Path of shuffling array.")
    parser.add_argument("--nb_permutations", type=str, default=None, help="Path of shuffling array.")
    parser.add_argument("--alpha_percentile", type=str, default=None, help="Path of shuffling array.")
    parser.add_argument("--login", type=str, default='', help="Login to connect to the cluster.")
    parser.add_argument("--password", type=str, default='', help="Password to connect to the cluster.")

    args = parser.parse_args()

    ### FileTransfers
    # FileTransfer creation for input/output files
    scripts_directory = FileTransfer(is_input=True,
                        client_path=scripts_path,
                        name="working directory")

    files_list = sorted(glob.glob(os.path.join(args.output, 'run_*_alpha_*.yml')))
    group_significativity = []
    jobs = []

    for yaml_file in files_list:
        info = os.path.basename(yaml_file).split('_')
        run = int(info[1])
        alpha = int(info[3])
        job = Job(command=["python", "significance_clusterized.py", 
                            "--yaml_file", yaml_file, 
                            "--output_r2", os.path.join(args.output, 'r2'), 
                            "--output_distribution", os.path.join(args.output, 'distribution'), 
                            "--x", args.x, 
                            "--y", args.y, 
                            "--shuffling", args.shuffling, 
                            "--n_permutations", args.nb_permutations, 
                            "--alpha_percentile", args.alpha_percentile], 
                    name="job {} - alpha {}".format(run, alpha), 
                    referenced_input_files=[scripts_directory], 
                    working_directory=scripts_directory)
        group_significativity.append(job)
        jobs.append(job)

    distribution_voxels = Group(elements=group_significativity,
                            name="Voxel wise fitting of the models")

    
    workflow2 = Workflow(jobs=jobs,
                    root_group=[distribution_voxels])


    ### Submit the workflow to computing resource (configured in the client-server mode)

    controller2 = WorkflowController("DSV_cluster_ap259944", args.login, args.password)

    workflow_id2 = controller2.submit_workflow(workflow=workflow2,
                                            name="Voxel-wise computations")
    
    # You may use the gui or manually transfer the files:
    manual = True
    if manual:
        Helper.transfer_input_files(workflow_id2, controller2)
        Helper.wait_workflow(workflow_id2, controller2)
        Helper.transfer_output_files(workflow_id2, controller2)