#!/usr/bin/env python3

import os
import numpy as np
import glob
import pandas as pd
import yaml
import argparse


# Functions definition
def write(path, text, end='\n'):
    """Write in text file at 'path'."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)

def delete_file(path):
    """Safe delete a file."""
    if os.path.isfile(path):
        os.system(f"rm {path}")


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nCreate the command line for the fourth part of the ridge analysis.""")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model.")
    parser.add_argument("--subject", type=str, default=None, help="Subject name.")
    parser.add_argument("--language", type=str, default=None, help="Subject name.")

    args = parser.parse_args()

    subject = args.subject
    model_name = args.model_name
    language = args.language

    # Define paths and variables
    inputs_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
    derivatives_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/")
    shuffling_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/shuffling.npy")
    yaml_files_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/yaml_files/")
    design_matrices_path = os.path.join(inputs_path, f"derivatives/fMRI/design-matrices/{language}/{model_name}/")
    fmri_path = os.path.join(inputs_path, f"data/fMRI/{language}/{subject}/func/")
    jobs_state_folder = os.path.join(inputs_path, "command_lines/jobs_state/")
    parameters_path = os.path.join(derivatives_path, 'parameters.yml')

    path4model_subject = os.path.join(inputs_path, f"command_lines/4_{subject}_{model_name}_{language}.sh")
    delete_file(path4model_subject)
    
    # Retrieve data
    with open(parameters_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print(-1)
            quit()
    model = parameters['models'][0]
    name = model['name']
    nb_voxels = parameters['nb_voxels']
    alpha_percentile = parameters['alpha_percentile']
    nb_permutations = parameters['nb_permutations']
    nb_runs = parameters['nb_runs']
    check_before =  f"echo '1-->0' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')}"
    check_after =  f"echo '~0' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')} "
    write(path4model_subject, "#!/bin/sh")
    write(path4model_subject, check_before)

    # Create bash script for this model and subject
    command_merge = f"python {os.path.join(inputs_path, 'code/utilities/optimized_merge_results.py')} --input_folder {derivatives_path} " + \
                                    f"--nb_runs {nb_runs} " + \
                                    f"--yaml_files {yaml_files_path} " + \
                                    f"--nb_voxels {nb_voxels} " + \
                                    f"--n_permutations {nb_permutations} " + \
                                    f"--alpha_percentile {alpha_percentile} " + \
                                    f"--model_name '{name}' "

    command_create_maps = f"python {os.path.join(inputs_path, 'code/utilities/create_maps.py')} --input {derivatives_path} " + \
                                    f"--parameters {parameters_path} " + \
                                    f"--subject {subject} " + \
                                    f"--fmri_data {fmri_path} " 
        
    write(path4model_subject, command_merge)
    write(path4model_subject, command_create_maps)
    write(path4model_subject, check_after)
