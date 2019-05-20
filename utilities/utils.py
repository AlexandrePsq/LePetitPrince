import glob
import os
from os.path import join
from .settings import Paths, Extensions

from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
from sklearn.model_selection import LeaveOneGroupOut
import nibabel as nib
import numpy as np
import csv
from sklearn.metrics import r2_score


paths = Paths()
extensions = Extensions()


#########################################
############ Data retrieving ############
#########################################

def get_data(language, data_type, subject=None, source='', model='', test=False):
    # General function for data retrieving
    # Output: list of path to the different data files
    extension = extensions.get_extension(data_type)
    sub_dir = os.listdir(paths.path2data)
    if data_type in sub_dir:
        base_path = paths.path2data
        if data_type in ['fMRI', 'MEG']:
            file_pattern = '{2}/func/{0}_{1}_{2}'.format(data_type, language, subject) + '_run*' + extension
        else:
            file_pattern = '{}_{}'.format(data_type, language) + '_run*' + extension
    else:
        base_path = join(paths.path2derivatives, source)
        file_pattern = '{}_{}'.format(data_type, language) + '_' + model + '_run*' + extension
    if test:
        data = [join(base_path, '{0}/{1}/test/{2}'.format(data_type, language, data_type)+extension)]
    else:
        data = [sorted(glob.glob(join(base_path, '{0}/{1}/{2}'.format(data_type, language, model), file_pattern)))]
    return data


def get_output_parent_folder(source, output_data_type, language, model):
    return join(paths.path2derivatives, '{0}/{1}/{2}/{3}'.format(source, output_data_type, language, model))


def get_path2output(output_parent_folder, output_data_type, language, model, run_name, extension):
    return join(output_parent_folder, '{0}_{1}_{2}_{3}'.format(output_data_type, language, model, run_name) + extension)


def withdram(dataframe, run, run_indexes_dict):
    """Return the dataframe made of the runs data concatenation without the specified run."""
    beg, end = run_indexes_dict[run]
    result = np.vstack([dataframe[:beg, :], dataframe[end:,:]])
    return result



#########################################
###### Computation functionalities ######
#########################################

def compute(path, overwrite=False):
    # Tell us if we can compute or not
    result = True
    if os.path.isfile(path):
        result = overwrite
    return result


def check_folder(path):
    # Create adequate folders if necessary
    if not os.path.isdir(path):
        os.mkdir(path)



#########################################
################ Figures ################
#########################################

def create_maps(masker, distribution, distribution_name, subject, output_parent_folder):
    model = op.basename(output_parent_folder)
    language = op.basename(op.dirname(output_parent_folder))
    data_type = op.basename(op.dirname(op.dirname(output_parent_folder)))

    img = masker.inverse_transform(distribution)
    
    path2output_raw = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, language, model, distribution_name, subject)+'.nii.gz')
    path2output_png = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, language, model, distribution_name, subject)+'.png')

    nib.save(img, path2output_raw)

    display = plot_glass_brain(img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(path2output_png)
    display.close()



#########################################
################## Log ##################
#########################################

def log(subject, voxel, alpha, r2):
    """ log stats per fold to a csv file """
    logcsvwriter = csv.writer(open("test.log", "a+"))
    if voxel = 'whole brain':
        logcsvwriter.writerow([subject, voxel, np.mean(r2), np.std(r2),
                            np.min(r2), np.max(r2)])
    else:
        logcsvwriter.writerow([subject, voxel, alpha, r2])


#########################################
########## Classical functions ##########
#########################################


def get_r2_score(model, y_true, y2predict, r2_min=0., r2_max=0.99):
    # return the R2_score for each voxel (=list)
    r2 = r2_score(y_true,
                    model.predict(y2predict),
                    multioutput='raw_values')
    rsquares_training = clean_rscores(r2, r2_min, r2_max)
    # remove values with are too low and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2min or x >= r2max) else x for x in rscores])


def transform_design_matrices(path):
    # Read design matrice csv file and add a column with only 1
    dm = pd.read_csv(path, header=0)
    # add the constant
    const = np.ones((dm.shape[0], 1))
    dm = np.hstack((dm, const))
    return dm 



#########################################
######### First level analysis ##########
#########################################

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker


def do_single_subject(subject, fmri_runs, matrices, masker, output_parent_folder, model, voxel_wised=False):
    # Compute r2 maps for all subject for a given model
    #   - subject : e.g. : 'sub-060'
    #   - fmri_runs: list of fMRI data runs (1 for each run)
    #   - matrices: list of design matrices (1 for each run)
    #   - masker: MultiNiftiMasker object
    fmri_runs = [masker.transform(f) for f in fmri_filenames] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column

    # compute r2 maps and save them under .nii.gz and .png formats
    if voxel_wised:
        alphas, r2_test = per_voxel_analysis(model, fmri_runs, design_matrices, subject)
        create_maps(masker, alphas, 'alphas', subject, output_parent_folder) # alphas
    else:
        r2_test = whole_brain_analysis(model, fmri_runs, matrices, subject)
    create_maps(masker, r2_test, 'r2_test', subject, output_parent_folder) # r2 test


def whole_brain_analysis(model, fmri_runs, design_matrices, subject):
    #   - fmri_runs: list of fMRI data runs (1 for each run)
    #   - matrices: list of design matrices (1 for each run)
    r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    r2_test = None
    alpha = None

    logo = LeaveOneGroupOut() # leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(1, 10)):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train]) # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train = np.vstack([design_matrices[i] for i in train])
        model_fitted = model.fit(predictors_train, fmri_data_train)

        # return the R2_score for each voxel (=list)
        r2_test = get_r2_score(model_fitted, fmri_runs[test[0]], design_matrices[test[0]])

        # log the results
        log(subject, voxel='whole brain', alpha=None, r2=r2_test)

        scores = rsquares_test if r2_test is None else np.vstack([r2_test, r2_test])
    return np.mean(scores, axis=0) # compute mean vertically (in time)


    def per_voxel_analysis(model, fmri_runs, design_matrices, subject):
    # compute alphas and test score with cross validation
    #   - fmri_runs: list of fMRI data runs (1 for each run)
    #   - design_matrices: list of design matrices (1 for each run)
    #   - nb_voxels: number of voxels
    #   - indexes: dict specifying row indexes for each run
    alphas = np.zeros(nb_voxels)
    scores = np.zeros((nb_voxels, nb_runs))
    nb_voxels = fmri_runs[0].shape[1]
    nb_runs = len(fmri_runs)
    count = 0

    logo = LeaveOneGroupOut() # leave on run out !
    for train, test in logo.split(fmri_runs, groups=range(nb_runs)): # loop for r2 computation

        fmri_data_train = [fmri_runs[i] for i in train] # fmri_runs liste 2D colonne = voxels et chaque row = un t_i
        predictors_train = [design_matrices[i] for i in train]
        nb_samples = np.cumsum([0] + [fmri_data_train[i].shape[0] for i in range(len(fmri_data_train))]) # list of cumulative lenght
        indexes = {'run{}'.format(run+1): [nb_samples[i], nb_samples[i+1]] for i, run in enumerate(train)}

        model.cv = Splitter(indexes=indexes, n_splits=nb_runs)
        dm = np.vstack(predictors_train)
        fmri = np.vstack(fmri_data_train)

        for voxel in range(nb_voxels):
            X = dm[:,voxel].reshape((dm.shape[0],1))
            y = fmri[:,voxel].reshape((fmri.shape[0],1))
            # fit the model for a given voxel
            model_fitted = model.fit(X,y)
            # retrieve the best alpha and compute the r2 score for this voxel
            alphas[voxel] = model.alpha_
            scores[voxel, count] = get_r2_score(model_fitted, 
                                                fmri_runs[test[0]][:,voxel].reshape((fmri_runs[test[0]].shape[0],1)), 
                                                design_matrices[test[0]][:,voxel].reshape((design_matrices[test[0]].shape[0],1)))
            # log the results
            log(subject, voxel=voxel, alpha=model_fitted.alpha_, r2=r2_test)
        count += 1
        
    return alphas, np.mean(scores, axis=0) # compute mean vertically (in time)