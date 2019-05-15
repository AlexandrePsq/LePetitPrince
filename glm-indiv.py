############################################
# First level analysis 
# --> R2 maps
#
############################################


import argparse
from os.path import join

import warnings
warnings.simplefilter(action='ignore' )

from ..utilities.settings import Pathsn Subjects
from ..utilities.utils import *
import pandas as pd
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

paths = Paths()
subjects_list = Subjects()


def transform_design_matrices(path):
    dm = pd.read_csv(path, header=0)
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def compute_crossvalidated_r2(fmri_runs, design_matrices, loglabel, logcsvwriter):

    def log(r2_train, r2_test):
        """ log stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, 'training', np.mean(r2_train), np.std(r2_train),
                               np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(r2_test), np.std(r2_test),
                               np.min(r2_test), np.max(r2_test)])

    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None

    logo = LeaveOneGroupOut()#leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train]) # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train = np.vstack([design_matrices[i] for i in train])
        model = LinearRegression().fit(predictors_train, fmri_data_train)

        rsquares_training = r2_score(fmri_data_train,
                                     model.predict(predictors_train),
                                     multioutput='raw_values')
        rsquares_training = clean_rscores(rsquares_training, .0, .99)

        test_run = test[0]
        rsquares_test = r2_score(fmri_runs[test_run],
                                 model.predict(design_matrices[test_run]),
                                 multioutput='raw_values')
        rsquares_test = clean_rscores(rsquares_test, .0, .99)

        log(rsquares_training, rsquares_test)

        r2_train = rsquares_training if r2_train is None else np.vstack([r2_train, rsquares_training])
        r2_test = rsquares_test if r2_test is None else np.vstack([r2_test, rsquares_test])

    return (np.mean(r2_train, axis=0), np.mean(r2_test, axis=0))


def do_single_subject(rootdir, subj, matrices):
    fmri_filenames = get_fmri_files_from_subject(rootdir, subj)
    fmri_runs = [masker.transform(f) for f in fmri_filenames] # renvoie liste de matrices 2D contenant les valeurs des pixels dans le mask

    loglabel = subj
    logcsvwriter = csv.writer(open("test.log", "a+"))

    r2train, r2test = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)

    r2train_img = masker.inverse_transform(r2train)
    nib.save(r2train_img, f'train_{subj:03d}.nii.gz')

    r2test_img = masker.inverse_transform(r2test)
    nib.save(r2test_img, f'test_{subj:03d}.nii.gz')

    display = plot_glass_brain(r2test_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'test_{subj:03}.png')
    display.close()

    display = plot_glass_brain(r2train_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'train_{subj:03}.png')
    display.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='en', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")

    args = parser.parse_args()

    for model_ in args.models[0]:
        subjects = subjects_list.get_all(args.language)
        data_type = 'design-matrices'
        dm = get_data(args.language, data_type, model_, source='fMRI', args.test)
        fmri_runs = [get_data(args.language, 'fMRI', test=args.test, subject=subject) for subject in subjects]
    
        path2output_raw1 = join(output_parent_folder, "glm-indiv_{0}_{1}_{2}_{3}".format(args.language, model_, 'train', subject_id)+'.nii.gz')
        path2output_raw2 = join(output_parent_folder, "glm-indiv_{0}_{1}_{2}_{3}".format(args.language, model_, 'test', subject_id)+'.nii.gz')
        path2output_png1 = join(output_parent_folder, "glm-indiv_{0}_{1}_{2}_{3}".format(args.language, model_, 'train', subject_id)+'.png')
        path2output_png2 = join(output_parent_folder, "glm-indiv_{0}_{1}_{2}_{3}".format(args.language, model_, 'test', subject_id)+'.png')

        if compute(path2output_raw1, agrs.overwrite) && compute(path2output_raw2, agrs.overwrite):
            matrices = [transform_design_matrices(run) for run in dm]
            masker = compute_global_masker(fmri_runs)  # sloow
            if PARALLEL:
                    Parallel(n_jobs=-1)(delayed(do_single_subject)(ROOTDIR, sub, MATRICES) for sub in subjects)
                else:
                    for sub in subjects:
                        print(f'Processing subject {sub:03d}...')
                        do_single_subject(ROOTDIR, sub, MATRICES)')
