import glob
import os
from os.path import join
from .settings import Paths, Extensions

from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
import nibabel as nib
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

def create_r2_maps(masker, r2, stage, subject, output_parent_folder):
    model = op.basename(output_parent_folder)
    language = op.basename(op.dirname(output_parent_folder))
    data_type = op.basename(op.dirname(op.dirname(output_parent_folder)))

    r2_img = masker.inverse_transform(r2)
    path2output_raw = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, language, model, stage, subject)+'.nii.gz')
    path2output_png = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, language, model, stage, subject)+'.png')

    nib.save(r2_img, path2output_raw)

    display = plot_glass_brain(r2_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(path2output_png)
    display.close()


#########################################
################## Log ##################
#########################################

def log(subject, stage, r2):
    """ log stats per fold to a csv file """
    logcsvwriter = csv.writer(open("test.log", "a+"))
    logcsvwriter.writerow([subject, stage, np.mean(r2), np.std(r2),
                           np.min(r2), np.max(r2)])


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

def compute_global_masker(files): # [[path, path2], [path3, path4]]
    # return a MultiNiftiMasker object
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True) # return a object that transforms a 4D barin into a 2D matrix of voxel-time and can do the reverse action
    masker.fit()
    return masker

def do_single_subject(subject, fmri_runs, matrices, masker, output_parent_folder, voxel_wised=False):
    # Compute r2 maps for all subject for a given model
    #   - subject : e.g. : 'sub-060'
    #   - fmri_runs: list of fMRI data runs (1 for each run)
    #   - matrices: list of design matrices (1 for each run)
    #   - masker: MultiNiftiMasker object

    fmri_runs = [masker.transform(f) for f in fmri_filenames] # return a list of 2D matrices with the values of the voxels in the mask: 1 voxel per column

    # compute r2 maps and save them under .nii.gz and .png formats
    if voxel_wised:
        alphas, r2_train, r2_test = compute_alpha(model, fmri_runs, design_matrices, subject, voxel_wised)
    else:
        r2_train, r2_test = compute_crossvalidated_r2(model, fmri_runs, matrices, subject)
    create_r2_maps(masker, r2_train, 'train', subject, output_parent_folder) # r2 train
    create_r2_maps(masker, r2_test, 'test', subject, output_parent_folder) # r2 test