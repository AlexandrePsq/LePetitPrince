#!/usr/bin/env python3

import os
import numpy as np
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

    parser = argparse.ArgumentParser(description="""Objective:\nCreate the command line for the first part of the ridge analysis.""")
    parser.add_argument("--model_name", type=str, default='', help="Name of the model.")
    parser.add_argument("--subject", type=str, default='sub-057', help="Subject name.")
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

    path4model_subject = os.path.join(inputs_path, f"command_lines/1_{subject}_{model_name}_{language}.sh")
    paths_commands = [os.path.join(inputs_path, f"command_lines/1_{subject}_{model_name}_{language}_run{run}.sh") for run in range(1,10)]
    delete_file(path4model_subject)
    for path in paths_commands:
        delete_file(path)


    # Retrieve data
    with open(parameters_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print(-1)
            quit()
    model = parameters['models'][0]
    name = model['name']
    nb_runs = parameters['nb_runs']
    alphas = parameters['alphas']
    nb_features = parameters['nb_features']
    nb_permutations = parameters['nb_permutations']

    # Create bash script for this model and subject
    command = f"python {os.path.join(inputs_path, 'code/utilities/shuffling_preparation.py')} --nb_features {nb_features} --output {shuffling_path} --n_permutations {nb_permutations}"
    check_before =  f"echo '5-->4' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')}"
    check_after =  f"echo '~4' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')} "
    write(path4model_subject, "#!/bin/sh")
    write(path4model_subject, check_before)
    write(path4model_subject, command)
    write(path4model_subject, check_after)

    check_before =   f"echo '4-->3' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')}"
    check_after =  f"echo '~3' >  {os.path.join(jobs_state_folder, '+'.join([model_name, subject])+'_tmp.txt')} "
    write(paths_commands[0], check_before)
    for run in range(1, 1+int(nb_runs)):
        indexes = np.arange(1, 1+int(nb_runs))
        indexes_tmp = np.delete(indexes, run-1, 0)
        command = f"python {os.path.join(inputs_path, 'code/utilities/cv_alphas.py')} --indexes {','.join([str(i) for i in indexes_tmp])} " + \
                                        f"--output {yaml_files_path} " + \
                                        f"--x {design_matrices_path} " + \
                                        f"--y {fmri_path} " + \
                                        f"--run {run} " + \
                                        f"--alphas {alphas}"
        write(paths_commands[run-1], command)
    write(paths_commands[-1], check_after)
