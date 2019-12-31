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

    parser = argparse.ArgumentParser(description="""Objective:\nCreate the command line for the third part of the ridge analysis.""")
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
    fmri_path = os.path.join(inputs_path, f"data/fMRI/{language}/{subject}/func/")
    jobs_state_folder = os.path.join(inputs_path, "command_lines/jobs_state/")
    parameters_path = os.path.join(derivatives_path, 'parameters.yml')

    path4model_subject = os.path.join(inputs_path, f"command_lines/3_{subject}_{model_name}_{language}.sh")
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
    alpha_percentile = parameters['alpha_percentile']
    nb_permutations = parameters['nb_permutations']
    nb_runs = parameters['nb_runs']
    check_before =  f"echo '2-->1' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')}"
    check_after =  f"echo '~1' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')} "
    write(path4model_subject, "#!/bin/sh")
    write(path4model_subject, check_before)
    files_list = sorted(glob.glob(os.path.join(yaml_files_path, 'run_*_alpha_*.yml')))

    # Create bash script for this model and subject
    for file_ in files_list:
        run = os.path.basename(file_).split('_')[1]
        features_indexes = ','.join([str(index) for index in model['indexes']])
        command = f"python {os.path.join(inputs_path, 'code/utilities/optimized_generate_distribution.py')} --features_indexes {features_indexes} " + \
                                        f"--output {derivatives_path} " + \
                                        f"--yaml_file {file_} " + \
                                        f"--y {fmri_path} " + \
                                        f"--n_sample {nb_permutations} " + \
                                        f"--shuffling {shuffling_path} " + \
                                        f"--model_name '{name}' " + \
                                        f"--paths2model {args.path2models} "
        write(path4model_subject, command)
    write(path4model_subject, check_after)
    