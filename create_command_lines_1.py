import os
import numpy as np
import pandas as pd
import yaml


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



# General parameters
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
subjects2run = ['sub-063', 'sub-067', 'sub-073', 'sub-077', 'sub-082', 'sub-101', 'sub-109', 'sub-110']
language = "english"
alpha_percentile = str(99)
nb_permutations = 3000
job2launch = []

if __name__=='__main__':

    # Create bash script for each model and each subject
    for subject in subjects2run:
        for model_name in model_names:
            shuffling_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/shuffling.npy"
            yaml_files_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/yaml_files/"
            design_matrices_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/design-matrices/{language}/{model_name}/"
            fmri_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI/{language}/{subject}/func/" 
            y = np.load(os.path.join(fmri_path, 'y_run1.npy'))
            x = np.load(os.path.join(design_matrices_path, 'x_run1.npy'))
            nb_voxels = str(y.shape[1])
            nb_features = str(x.shape[1])
            alpha_list = [round(tmp, 5) for tmp in np.logspace(2, 5, 25)]
            alphas = ','.join([str(alpha) for alpha in alpha_list]) 
            nb_runs = 9

            # Defining 'parameters.yml'file
            parameters_path = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/ridge-indiv/{language}/{subject}/{model_name}/parameters.yml"
            parameters = {'models': [], 
                            'nb_runs':nb_runs, 
                            'nb_voxels':nb_voxels, 
                            'nb_features':nb_features, 
                            'nb_permutations':nb_permutations, 
                            'alphas':alphas,
                            'alpha_percentile': alpha_percentile}
            parameters['models'].append({'name':'', 'indexes':[0, int(nb_features)]})

            with open(parameters_path, 'w') as outfile:
                yaml.dump(parameters, outfile, default_flow_style=False)


            command = f"python shuffling_preparation.py --nb_features {nb_features} --output {shuffling_path} --n_permutations {nb_permutations}"
            path4model_subject = f"/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/1_{subject}_{model_name}_{language}.sh"
            job2launch.append(path4model_subject)
            check_folder(os.path.dirname(path4model_subject))
            write(path4model_subject, "#!/bin/sh")
            write(path4model_subject, command)

            
            for run in range(1, 1+int(nb_runs)):
                indexes = np.arange(1, 1+int(nb_runs))
                indexes_tmp = np.delete(indexes, run-1, 0)
                command = f"python cv_alphas.py --indexes {','.join([str(i) for i in indexes_tmp])} " + \
                                                f"--output {yaml_files_path} " + \
                                                f"--x {design_matrices_path} " + \
                                                f"--y {fmri_path} " + \
                                                f"--run {run} " + \
                                                f"--alphas {alphas}"
                write(path4model_subject, command)

    job2launchpath = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/command_lines/job2launch.txt"
    jobs_state = pd.DataFrame(data=np.full((len(job2launch),4), np.nan)  , columns=['subject', 'model', 'name', 'state'])
    for index, job in enumerate(job2launch):
        queue='Nspin_bigM' if 'all-layers' in job else 'Nspin_long' 
        walltime='80:00:00'
        output_log=f'/home/ap259944/logs/log_o_{job}'
        error_log=f'/home/ap259944/logs/log_e_{job}'
        job_name=f'{job}'
        write(job2launchpath, f"qsub -q {queue} -N {job_name} -l walltime={walltime} -o {output_log} -e {error_log} {job}")
        jobs_state.iloc[index] = [os.path.basename(job).split('_')[1], os.path.basename(job).split('_')[2], job, 0]
    jobs_state =  jobs_state.astype({'state':int}) 

