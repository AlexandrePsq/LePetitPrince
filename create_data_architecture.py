################################################################################
# Create the local folder architecture necessary for the script to run correctly
# Should be run from: $LePetitPrince/code/
#       $ cd $LePetitPrince/code/
#       $ python create_data_architecture.py
################################################################################

import os
from utilities.settings import Subjects, Paths, Params
from utilities.utils import check_folder

subjects = Subjects()
paths = Paths()
params = Params()
languages = params.languages


##################### ROOT #####################
os.chdir(paths.path2root)
check_folder('data')
check_folder('derivatives')
check_folder('paradigm')
check_folder('oldstuff')


##################### DATA #####################
os.chdir(paths.path2data)
## fMRI ##
check_folder('fMRI')
for language in languages:
    check_folder('fMRI/{}'.format(language))
    for subject in subjects.get_all(language):
        check_folder('fMRI/{}/{}'.format(language, subject))
        check_folder('fMRI/{}/{}/anat'.format(language, subject))
        check_folder('fMRI/{}/{}/func'.format(language, subject))

## MEG ##
check_folder('MEG')
# to be done

## TEXT ##
check_folder('text')
for language in languages:
    check_folder('text/{}'.format(language))
    for model in params.models:
        check_folder('text/{}/{}'.format(language, model))
        check_folder('text/{}/{}/onsets-offsets'.format(language, model))
    # LePetitPrince raw text and chapters should be put in text/$language

## WAVE ##
check_folder('wave')
for language in languages:
    check_folder('wave/{}'.format(language))
    # to be done




################## DERIVATIVES ##################
os.chdir(paths.path2derivatives)
check_folder('fMRI')
folders = ['raw-features', 'features', 'design-matrices', 'glm-indiv', 'glm-group', 'ridge-indiv', 'ridge-group']
for folder in folders:
    check_folder(os.path.join('fMRI', folder))
    for language in languages:
        check_folder(os.path.join('fMRI', folder, language))
    

check_folder('MEG')
# to be done


os.chdir(paths.path2code)