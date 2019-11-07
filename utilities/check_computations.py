import os
import pandas as pd
import argparse
import warnings
warnings.simplefilter(action='ignore')
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
import glob
import yaml
import numpy as np
import time


def delete_file(path):
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
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, smoothing_fwhm=5) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker



model_names = ['wordrate_word_position',
                'wordrate_model',
                'wordrate_log_word_freq',
                'wordrate_function-word',
                'wordrate_content-word',
                'wordrate_all_model',
                'topdown_model',
                'rms_model',
                'other_sentence_onset',
                'mfcc_model',
                'gpt2_layer-9',
                'gpt2_layer-8',
                'gpt2_layer-7',
                'gpt2_layer-6',
                'gpt2_layer-5',
                'gpt2_layer-4',
                'gpt2_layer-3',
                'gpt2_layer-2',
                'gpt2_layer-12',
                'gpt2_layer-11',
                'gpt2_layer-10',
                'gpt2_layer-1',
                'gpt2_embeddings',
                'gpt2_all-layers',
                'bottomup_model',
                'bert_bucket_median_all-layers',
                'bert_bucket_layer-9',
                'bert_bucket_layer-8',
                'bert_bucket_layer-7',
                'bert_bucket_layer-6',
                'bert_bucket_layer-5',
                'bert_bucket_layer-4',
                'bert_bucket_layer-3',
                'bert_bucket_layer-2',
                'bert_bucket_layer-12',
                'bert_bucket_layer-11',
                'bert_bucket_layer-10',
                'bert_bucket_layer-1',
                'bert_bucket_embeddings',
                'bert_bucket_all-layers']
subjects = ['sub-057', 'sub-063', 'sub-067', 'sub-073', 'sub-077', 'sub-082', 'sub-101', 'sub-109', 'sub-110', 'sub-113', 'sub-114']
nb_runs = 9
language = 'english'
nb_permutations = 3000
alpha_percentile = str(99)
job2launch1 = []
job2launch2 = []
job2launch3 = []
job2launch4 = []

job2launchpath1 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch1.txt"
job2launchpath2 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch2.txt"
job2launchpath3 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch3.txt"
job2launchpath4 = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch4.txt"


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nScheduler that checks the state of all the jobs and planned the jobs that need to be run.""")
    parser.add_argument("--jobs_state_folder", type=str, default="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/jobs_state/", help="Folder where jobs state are saved.")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Overwrite existing data.")

    args = parser.parse_args()

    inputs_path = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/"
    jobs_state_path = os.path.join(inputs_path, "command_lines/jobs_state.csv")
    jobs_state_folder = os.path.join(inputs_path, "command_lines/jobs_state/")

    print('Computing global masker...', end=' ', flush=True)
    fmri_runs = {}
    for subject in subjects:
        fmri_path = os.path.join(inputs_path, "data/fMRI/{language}/{subject}/func/")
        check_folder(fmri_path)
        fmri_runs[subject] = sorted(glob.glob(os.path.join(fmri_path.format(language=language, subject=subject), 'fMRI_*run*')))
    masker = compute_global_masker(list(fmri_runs.values()))

    jobs_state = pd.DataFrame(data=np.full((len(model_names)*len(subjects),3), np.nan)  , columns=['subject', 'model', 'state'])
    index = 0
    print('--> Done', flush=True)
    print('Iterating over subjects...\nTransforming fMRI data...', flush=True)
    for subject in subjects:
        print('\tSubject: {} ...'.format(subject), end=' ', flush=True)
        fmri_path = os.path.join(inputs_path, f"data/fMRI/{language}/{subject}/func/")
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
        for model_name in model_names:

            ######################
            ### Data retrieval ###
            ######################
            derivatives_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/")
            shuffling_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/shuffling.npy")
            r2_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/r2/")
            pearson_corr_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/pearson_corr/")
            distribution_r2_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/distribution_r2/")
            distribution_pearson_corr_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/distribution_pearson_corr/")
            yaml_files_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/yaml_files/")
            output_path = os.path.join(inputs_path, f"derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/outputs/")
            design_matrices_path = os.path.join(inputs_path, f"derivatives/fMRI/design-matrices/{language}/{model_name}/")
            parameters_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/parameters.yml"
            
            ####################
            ### Sanity check ###
            ####################

            all_paths = [inputs_path, design_matrices_path, derivatives_path, 
                            r2_path, pearson_corr_path, distribution_r2_path, 
                            distribution_pearson_corr_path, yaml_files_path, output_path,
                            jobs_state_folder, os.path.dirname(parameters_path)]
            for path in all_paths:
                check_folder(path)

            if not os.path.isfile(parameters_path) or args.overwrite:
                y = np.load(os.path.join(fmri_path, 'y_run1.npy'))
                x = np.load(os.path.join(design_matrices_path, 'x_run1.npy'))
                nb_voxels = str(y.shape[1])
                nb_features = str(x.shape[1])
                alpha_list = [round(tmp, 5) for tmp in np.logspace(2, 5, 25)]
                alphas = ','.join([str(alpha) for alpha in alpha_list]) 
                # Defining 'parameters.yml' file
                parameters = {'models': [], 
                                'nb_runs':nb_runs, 
                                'nb_voxels':nb_voxels, 
                                'nb_features':nb_features, 
                                'nb_permutations':nb_permutations, 
                                'alphas':alphas,
                                'alpha_percentile': alpha_percentile}
                parameters['models'].append({'name':'', 'indexes':[0, int(nb_features)]})
                # Saving
                with open(parameters_path, 'w') as outfile:
                    yaml.dump(parameters, outfile, default_flow_style=False)
                    print('\t\tParameters for model {} is Done.'.format(model_name))

            if not os.path.isfile(jobs_state_path):
                jobs_state.iloc[index] = [subject, model_name, '5']
                index += 1
    
    if not os.path.isfile(jobs_state_path) or args.overwrite:
        jobs_state.to_csv(jobs_state_path, index=False)
    else:
        jobs_state = pd.read_csv(jobs_state_path)

    state_files = sorted(glob.glob(os.path.join(args.jobs_state_folder, '*.txt')))

    ################ INFINITE LOOP ################
    print('---------------------- Ridge pipeline scheduler is on... ----------------------')
    print('--------------------------- (Iteration every 15 min)---------------------------')
    try:
        while True:
            delete_file(job2launchpath1)
            delete_file(job2launchpath2)
            delete_file(job2launchpath3)
            delete_file(job2launchpath4)
            # Update state CSV
            for state_file in state_files:
                model_name = os.path.basename(state_file).split('_')[0]
                subject = os.path.basename(state_file).split('_')[1].split('.')[0]
                state = open(state_file, 'r').read()
                derivatives_path = os.path.join(inputs_path, "derivatives/fMRI/ridge-indiv/english/{}/{}/".format(subject, model_name))
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
                    result = (len(glob.glob(os.path.join(derivatives_path, 'outputs/maps/*')))==39)
                else:
                    state = '~' + state
                    result = True
                state = state[1:] if result else str(int(state[1:])+1)
                sel = (jobs_state.subject == subject) & (jobs_state.model_name == model_name)
                jobs_state.loc[sel, 'state'] = state
                os.system(f"rm {state_file}")
            
            # List new jobs to run
            for state_file in state_files:
                model_name = os.path.basename(state_file).split('_')[0]
                subject = os.path.basename(state_file).split('_')[1].split('.')[0]
                state = open(state_file, 'r').read()
                derivatives_path = os.path.join(inputs_path, "derivatives/fMRI/ridge-indiv/english/{}/{}/".format(subject, model_name))
                if state in ['5', '4']:
                    os.system(f"python {os.path.join(derivatives_path, 'code/create_command_lines_1.py')} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch1.append(os.path.join(inputs_path, f"command_lines/1_{subject}_{model_name}_{language}.sh"))
                elif state=='3':
                    os.system(f"python {os.path.join(derivatives_path, 'code/create_command_lines_2.py')} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch2.append(os.path.join(inputs_path, f"command_lines/2_{subject}_{model_name}_{language}.sh"))
                elif state=='2':
                    os.system(f"python {os.path.join(derivatives_path, 'code/create_command_lines_3.py')} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch3.append(os.path.join(inputs_path, f"command_lines/3_{subject}_{model_name}_{language}.sh"))
                elif state=='1':
                    os.system(f"python {os.path.join(derivatives_path, 'code/create_command_lines_4.py')} --model_name {model_name} --subject {subject} --language {language}")
                    job2launch4.append(os.path.join(inputs_path, f"command_lines/4_{subject}_{model_name}_{language}.sh"))



            for index, job in enumerate(job2launch1):
                queue='Nspin_bigM' if 'all-layers' in job else ('Nspin_long' if (('bert' in job) or ('gpt2' in job) or ('lstm' in job)) else 'Nspin_short')
                walltime='80:00:00' if queue in ['Nspin_bigM', 'Nspin_long'] else '01:00:00'
                output_log=f'/home/ap259944/logs/log_o_{job}'
                error_log=f'/home/ap259944/logs/log_e_{job}'
                job_name=f'{job}'
                write(job2launchpath1, f"qsub -q {queue} -N {os.path.basename(job_name).split('.')[0]} -l walltime={walltime} -o {output_log} -e {error_log} {job}")

            for index, job in enumerate(job2launch2):
                queue='Nspin_bigM' if 'all-layers' in job else ('Nspin_long' if (('bert' in job) or ('gpt2' in job) or ('lstm' in job)) else 'Nspin_short')
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

            time.sleep(900)
    except:
        print('---------------------- Ridge pipeline scheduler is off. ----------------------')