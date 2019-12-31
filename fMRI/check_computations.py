import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

    
import pandas as pd
import argparse
import warnings
warnings.simplefilter(action='ignore')
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from utilities.utils import get_design_matrix, read_yaml
from utilities.settings import Paths
import glob
from tqdm import tqdm
import yaml
import numpy as np
import time


def delete_file(path):
    """Safe delete a file."""
    if os.path.isfile(path):
        os.system(f"rm {path}")

# Functions definition
def write(path, text):
    """Write in text file at 'path'."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write('\n')

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    """Returns a MultiNiftiMasker object."""
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def retrieve_df(jobs_state, inputs_path, compute_distribution=False):
    print('Retrieving state CSV...', end=' ', flush=True)
    for index, row in tqdm(jobs_state.iterrows()):
        model = row.model_name
        sub = row.subject
        state = row.state
        derivatives_path = os.path.join(inputs_path, "derivatives/fMRI/ridge-indiv/english/{}/{}/".format(sub, model))
        if (len(glob.glob(os.path.join(derivatives_path, 'outputs/maps/*')))==9):
            state = '0'
        else:
            if os.path.isfile(os.path.join(derivatives_path, 'shuffling.npy')):
                result = [os.path.isfile(os.path.join(derivatives_path, file_)) for file_ in ['yaml_files/voxel2alpha1.npy', 'yaml_files/voxel2alpha2.npy', 'yaml_files/voxel2alpha3.npy', 'yaml_files/voxel2alpha4.npy', 'yaml_files/voxel2alpha5.npy', 'yaml_files/voxel2alpha6.npy', 'yaml_files/voxel2alpha7.npy', 'yaml_files/voxel2alpha8.npy', 'yaml_files/voxel2alpha9.npy']]
                if (len(result)==sum(result)):
                    yaml_files = sorted(glob.glob(os.path.join(derivatives_path, 'yaml_files/*.yml')))
                    r2_files = sorted(glob.glob(os.path.join(derivatives_path, 'r2/*.npy')))
                    pearson_files = sorted(glob.glob(os.path.join(derivatives_path, 'pearson_corr/*.npy')))
                    if ((len(yaml_files)==len(r2_files)) and (len(yaml_files)==len(pearson_files))):
                        distribution_r2_files = sorted(glob.glob(os.path.join(derivatives_path, 'distribution_r2/*.npy')))
                        distribution_pearson_corr_files = sorted(glob.glob(os.path.join(derivatives_path, 'distribution_pearson_corr/*.npy')))
                        if ((len(yaml_files)==len(distribution_r2_files)) and (len(yaml_files)==len(distribution_pearson_corr_files))):
                            state = '1'
                        else:
                            state = '2' if compute_distribution else '1'
                    else:
                        state = '3'
                else:
                    state = '4'
            else:
                state = '5'
        sel = (jobs_state.subject == sub) & (jobs_state.model_name == model)
        jobs_state.loc[sel, 'state'] = str(state)
    print('\t--> Done')
    return jobs_state


job2launchpath1 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch1.txt"
job2launchpath2 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch2.txt"
job2launchpath3 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch3.txt"
job2launchpath4 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch4.txt"

paths = Paths()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nScheduler that checks the state of all the jobs and planned the jobs that need to be run.""")
    parser.add_argument("--jobs_state_folder", type=str, default="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/jobs_state/", help="Folder where jobs state are saved.")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Overwrite existing data.")
    parser.add_argument("--subjects", nargs='+', action='append', help="Id of the subjects.")
    parser.add_argument("--path2models", type=str, default=None, help="Path were the aggregated models are specified.")
    parser.add_argument("--path2parameters", type=str, default=None, help="Path were the parameters.")
    parser.add_argument("--compute_distribution", action="store_true", default=False, help="allow the computation of predictions over the randomly shuffled columns of the test set")

    args = parser.parse_args()
    subjects = args.subjects[0]
    data = read_yaml(args.path2models)
    shared_parameters = read_yaml(args.path2parameters)
    language = data['aggregated_models']['language']
    nb_runs = shared_parameters['nb_runs']
    alpha_percentile = shared_parameters['alpha_percentile']
    all_models = data['aggregated_models']['all_models']
    model_names = all_models.keys()
    nb_permutations = shared_parameters['nb_permutations']
    alpha_list = [round(tmp, 5) for tmp in np.logspace(shared_parameters['alpha_min_log_scale'], shared_parameters['alpha_max_log_scale'], shared_parameters['nb_alphas'])]
    alphas = ','.join([str(alpha) for alpha in alpha_list]) 

    jobs_state_path = os.path.join(paths.path2root, "command_lines/jobs_state.csv")
    jobs_state_folder = args.jobs_state_folder

    if args.overwrite:
        print('Computing global masker...', end=' ', flush=True)
        fmri_runs = {}
        for subject in subjects:
            fmri_path = os.path.join(paths.path2data, "/fMRI/{language}/{subject}/func/")
            check_folder(fmri_path)
            fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path.format(language=language, subject=subject), 'fMRI_*run*')))
        masker = compute_global_masker(list(fmri_runs.values()))

        jobs_state = pd.DataFrame(data=np.full((len(all_models)*len(subjects),3), np.nan)  , columns=['subject', 'model_name', 'state'])
        index_jobs_state = 0
        print('--> Done', flush=True)
    else:
        jobs_state = pd.read_csv(jobs_state_path)

    print('Iterating over subjects...\nTransforming fMRI data...', flush=True)
    for subject in subjects:
        print('\tSubject: {} ...'.format(subject), end=' ', flush=True)
        fmri_path = os.path.join(paths.path2root, f"data/fMRI/{language}/{subject}/func/")
        check_folder(fmri_path)
        runs = sorted(glob.glob(os.path.join(fmri_path, 'fMRI_*run*')))

        if len(glob.glob(os.path.join(fmri_path, 'y_run*.npy'))) != 9 or args.overwrite:

            ###########################
            ### Data transformation ###
            ###########################

            y = [masker.transform(f) for f in runs] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column
            # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
            for run in range(len(y)):
                new = np.random.random(y[run].shape[0])/10000
                zero = np.zeros(y[run].shape[0])
                y[run] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, y[run])

            for index in range(len(y)):
                np.save(os.path.join(fmri_path, 'y_run{}.npy'.format(index+1)), y[index])
            print('(dim: {})'.format(y[0].shape), end=' ', flush=True)
        print('--> Done', flush=True)
        for key in all_models.keys():
            model_name = key
            ######################
            ### Data retrieval ###
            ######################
            derivatives_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/")
            shuffling_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/shuffling.npy")
            r2_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/r2/")
            pearson_corr_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/pearson_corr/")
            distribution_r2_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/distribution_r2/")
            distribution_pearson_corr_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/distribution_pearson_corr/")
            yaml_files_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/yaml_files/")
            output_path = os.path.join(paths.path2root, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/outputs/")
            design_matrices_path = os.path.join(paths.path2root, f"derivatives/fMRI/design-matrices/{language}/{model_name}/")
            parameters_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/parameters.yml"
            
            ####################
            ### Sanity check ###
            ####################

            all_paths = [paths.path2root, design_matrices_path, derivatives_path, 
                            r2_path, pearson_corr_path, distribution_r2_path, 
                            distribution_pearson_corr_path, yaml_files_path, output_path,
                            jobs_state_folder, os.path.dirname(parameters_path)]
            for path in all_paths:
                check_folder(path)
            
            if not os.path.isfile(parameters_path) or args.overwrite:
                y = np.load(os.path.join(fmri_path, 'y_run1.npy'))
                x = get_design_matrix(all_models[key], language)[0]
                nb_voxels = str(y.shape[1])
                nb_features = str(x.shape[1])
                # Defining 'parameters.yml' file
                parameters = {'models': [], 
                                'nb_runs':nb_runs, 
                                'nb_voxels':nb_voxels, 
                                'nb_features':nb_features, 
                                'nb_permutations':nb_permutations, 
                                'alphas':alphas,
                                'alpha_percentile': alpha_percentile,
                                'tr': shared_parameters['tr'],
                                'scaling_mean': shared_parameters['scaling_mean'],
                                'scaling_var': shared_parameters['scaling_var'],
                                'parallel': shared_parameters['parallel'],
                                'cuda': shared_parameters['cuda'],
                                'voxel_wise': shared_parameters['voxel_wise'],
                                'atlas': shared_parameters['atlas'],
                                'seed': shared_parameters['seed'],
                                'alpha_min_log_scale': shared_parameters['alpha_min_log_scale'],
                                'alpha_max_log_scale': shared_parameters['alpha_max_log_scale'],
                                'nb_alphas': shared_parameters['nb_alphas']
                                }

                parameters['models'].append({'name':'', 'indexes':[0, int(nb_features)]})
                # Saving
                with open(parameters_path, 'w') as outfile:
                    yaml.dump(parameters, outfile, default_flow_style=False)

            if not os.path.isfile(jobs_state_path) or args.overwrite:
                jobs_state.iloc[index_jobs_state] = [subject, model_name, '5']
                index_jobs_state += 1
            else:
                if jobs_state[(jobs_state['subject']==subject) & (jobs_state['model_name']==model_name)].empty:
                    jobs_state.loc[len(jobs_state)] = [subject, model_name, '5']   

        print('\t\tParameters for all models are computed.')
    if args.overwrite:
        jobs_state = retrieve_df(jobs_state, paths.path2root, args.compute_distribution)
    jobs_state.to_csv(jobs_state_path, index=False)
    
    ################ INFINITE LOOP ################
    print('---------------------- Ridge pipeline scheduler is on... ----------------------')
    print('--------------------------- (Iteration every 15 min)---------------------------')
    try:
        while True:
            state_files = sorted(glob.glob(os.path.join(args.jobs_state_folder, '*_tmp.txt')))
            job2launch1 = []
            job2launch2 = []
            job2launch3 = []
            job2launch4 = []
            delete_file(job2launchpath1)
            delete_file(job2launchpath2)
            delete_file(job2launchpath3)
            delete_file(job2launchpath4)
            print('Updating state CSV...', end=' ', flush=True)
            for state_file in state_files:
                model_name = '+'.join(os.path.basename(state_file).split('+')[:-1])
                subject = os.path.basename(state_file).split('+')[-1].split('_')[0]
                state = str(open(state_file, 'r').read()).replace('\n', '')
                derivatives_path = os.path.join(paths.path2root, "derivatives/fMRI/ridge-indiv/english/{}/{}/".format(subject, model_name))
                if state=='~4':
                    result = os.path.isfile(os.path.join(derivatives_path, 'shuffling.npy'))
                elif state=='~3':
                    result = [os.path.isfile(os.path.join(derivatives_path, file_)) for file_ in ['yaml_files/voxel2alpha1.npy', 'yaml_files/voxel2alpha2.npy', 'yaml_files/voxel2alpha3.npy', 'yaml_files/voxel2alpha4.npy', 'yaml_files/voxel2alpha5.npy', 'yaml_files/voxel2alpha6.npy', 'yaml_files/voxel2alpha7.npy', 'yaml_files/voxel2alpha8.npy', 'yaml_files/voxel2alpha9.npy']]
                    result = (len(result)==sum(result))
                elif state=='~2':
                    yaml_files = sorted(glob.glob(os.path.join(derivatives_path, 'yaml_files/*.yml')))
                    r2_files = sorted(glob.glob(os.path.join(derivatives_path, 'r2/*.npy')))
                    pearson_files = sorted(glob.glob(os.path.join(derivatives_path, 'pearson_corr/*.npy')))
                    result = ((len(yaml_files)==len(r2_files)) and (len(yaml_files)==len(pearson_files)))
                elif state=='~1':
                    yaml_files = sorted(glob.glob(os.path.join(derivatives_path, 'yaml_files/*.yml')))
                    distribution_r2_files = sorted(glob.glob(os.path.join(derivatives_path, 'distribution_r2/*.npy')))
                    distribution_pearson_corr_files = sorted(glob.glob(os.path.join(derivatives_path, 'distribution_pearson_corr/*.npy')))
                    result = ((len(yaml_files)==len(distribution_r2_files)) and (len(yaml_files)==len(distribution_pearson_corr_files)))
                elif state=='~0':
                    result = (len(glob.glob(os.path.join(derivatives_path, 'outputs/maps/*')))==9)
                else:
                    state = '~' + state
                    result = True
                state = state[1:] if result else str(int(state[1:])+1)
                sel = (jobs_state.subject == subject) & (jobs_state.model_name == model_name)
                jobs_state.loc[sel, 'state'] = str(state)
                if not (jobs_state[(jobs_state['subject']==subject) & (jobs_state['model_name']==model_name)].empty):
                    os.system(f"rm {state_file}")
            jobs_state.to_csv(jobs_state_path, index=False)
            print('\t--> Done')
            
            print('Listing new jobs to run...', flush=True)
            for index, row in tqdm(jobs_state.iterrows()):
                model_name = row['model_name']
                data['aggregated_models']['all_models']
                subject = row['subject']
                state = str(row['state'])
                if state in ['5', '4']:
                    os.system(f"python {os.path.join(paths.path2root, 'code/create_command_lines_1.py')} --path2models {args.path2models} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch1.append(os.path.join(paths.path2root, f"command_lines/1_{subject}_{model_name}_{language}.sh"))
                    #job2launch1 += [os.path.join(paths.path2root, f"command_lines/1_{subject}_{model_name}_{language}_run{run}.sh") for run in range(1,10)]
                elif state=='3':
                    os.system(f"python {os.path.join(paths.path2root, 'code/create_command_lines_2.py')} --path2models {args.path2models} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch2.append(os.path.join(paths.path2root, f"command_lines/2_{subject}_{model_name}_{language}.sh"))
                elif state=='2':
                    os.system(f"python {os.path.join(paths.path2root, 'code/create_command_lines_3.py')} --path2models {args.path2models} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch3.append(os.path.join(paths.path2root, f"command_lines/3_{subject}_{model_name}_{language}.sh"))
                elif state=='1':
                    os.system(f"python {os.path.join(paths.path2root, 'code/create_command_lines_4.py')} --path2models {args.path2models} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch4.append(os.path.join(paths.path2root, f"command_lines/4_{subject}_{model_name}_{language}.sh"))
            print('\t--> Done')

            print("Grouping 'qsub' commands...", end=' ', flush=True)
            for index, job in enumerate(job2launch1):
                queue='Nspin_bigM' if 'all-layers' in job else 'Nspin_long'
                walltime='80:00:00'
                output_log=f'/home/ap259944/logs/log_o_{job}'
                error_log=f'/home/ap259944/logs/log_e_{job}'
                job_name=f'{job}'
                write(job2launchpath1, f"qsub -q {queue} -N {os.path.basename(job_name).split('.')[0]} -l walltime={walltime} -o {output_log} -e {error_log} {job}")

            for index, job in enumerate(job2launch2):
                queue='Nspin_bigM' if (('bert' in job) or ('gpt2' in job) or ('lstm' in job)) else 'Nspin_long'
                walltime='99:00:00' if queue in ['Nspin_bigM', 'Nspin_long'] else '01:00:00'
                output_log=f'/home/ap259944/logs/log_o_{job}'
                error_log=f'/home/ap259944/logs/log_e_{job}'
                job_name=f'{job}'
                write(job2launchpath2, f"qsub -q {queue} -N {os.path.basename(job_name).split('.')[0]} -l walltime={walltime} -o {output_log} -e {error_log} {job}")

            for index, job in enumerate(job2launch3):
                queue='Nspin_bigM' if (('bert' in job) or ('gpt2' in job) or ('lstm' in job)) else 'Nspin_long'
                walltime='500:00:00'
                output_log=f'/home/ap259944/logs/log_o_{job}'
                error_log=f'/home/ap259944/logs/log_e_{job}'
                job_name=f'{job}'
                write(job2launchpath3, f"qsub -q {queue} -N {os.path.basename(job_name).split('.')[0]} -l walltime={walltime} -o {output_log} -e {error_log} {job}")

            for index, job in enumerate(job2launch4):
                queue='Nspin_short'
                walltime='01:00:00'
                output_log=f'/home/ap259944/logs/log_o_{job}'
                error_log=f'/home/ap259944/logs/log_e_{job}'
                job_name=f'{job}'
                write(job2launchpath4, f"qsub -q {queue} -N {os.path.basename(job_name).split('.')[0]} -l walltime={walltime} -o {output_log} -e {error_log} {job}")
            print('\t--> Done')
            print('\t\t.\n\t\t.\n\t\t.')
            time.sleep(900)
    except Exception as e:
        print('---------------------- Ridge pipeline scheduler is off. ----------------------')
        print(e)