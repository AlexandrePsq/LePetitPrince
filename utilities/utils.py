import glob
import os
from os.path import join
from .settings import Paths, Extensions

from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain
import nibabel as nib
import csv


paths = Paths()
extensions = Extensions()

#########################################
############ Data retrieving ############
#########################################

def get_data(language, data_type, source='', model='', test=False, subject):
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

def get_r2_maps(masker, r2, stage, subject, output_parent_folder):
    model = op.basename(output_parent_folder)
    language = op.basename(op.dirname(output_parent_folder))
    data_type = op.basename(op.dirname(op.dirname(output_parent_folder)))
    source = op.base_path(op.dirname(op.dirname(op.dirname(output_parent_folder))))

    r2_img = masker.inverse_transform(r2)
    path2output_raw = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, args.language, model_, stage, subject)+'.nii.gz')
    path2output_png = join(output_parent_folder, "{0}_{1}_{2}_{3}_{4}".format(data_type, args.language, model_, stage, subject)+'.png')

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