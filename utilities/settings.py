###################### PARAMETERS ######################
# 
# 
# Here, all the parameters can be changed.
# 
# 
# 
# 
# 
########################################################

from os.path import join

class Paths:
    def __init__(self):
        # Paths
        # self.path2root = '/neurospin/unicog/protocols/IRMf/LePetitPrince'
        self.path2root = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince'
        self.path2data = join(self.path2root, 'data')
        self.path2derivatives = join(self.path2root, 'derivatives')
        self.path2code = join(self.path2root, 'code')
        self.path2paradigm = join(self.path2root, 'paradigm')
        self.path2oldstuff = join(self.path2root, 'oldstuff')
        self.path2test = join(self.path2oldstuff, 'test')


class Extensions:
    def __init__(self):
        # Retrieve extensions
        self.extensions = {'wave':'.wav',
                           'text':'.txt',
                           'raw_features':'.csv',
                           'features':'.csv',
                           'fMRI':'.nii.nii',
                           'design-matrices':'.csv'
        }

    def get_extension(self, data_type):
        return self.extensions[data_type]
        
class Scans:
    def __init__(self):
        # number of scans per run
        self.nscans = {'en':{'run1':282,
                             'run2':298,
                             'run3':340,
                             'run4':303,
                             'run5':265,
                             'run6':343,
                             'run7':325,
                             'run8':292,
                             'run9':368,
        },
                       'fr':{'run1':309,
                             'run2':326,
                             'run3':354,
                             'run4':315,
                             'run5':293,
                             'run6':378,
                             'run7':332,
                             'run8':294,
                             'run9':336,
        }
        }
    def get_nscans(self, language, run):
                       return self.nscans[language][run]
        
class Subjects:
    def __init__(self):
        # Subjects
        self.subject_lists = {'en': [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115],
                              'fr': []
        }
        self.subject_test = {'en': [57],
                             'fr': []
        }
        
    def get_subject(self, subject_number):
        if subject_number < 10:
            return 'sub-00{}'.format(subject_number)
        elif subject_number < 100:
            return 'sub-0{}'.format(subject_number)
        else:
            return 'sub-{}'.format(subject_number)

    def get_all(self, language, test=False):
        result = []
        if test:
            sub_list = self.subject_test[language]
        else:
            sub_list = self.subject_lists[language]
        for subj in sub_list:
            result.append(self.get_subject(subj))
        return result

        
class Rois:
    def __init__(self):
        # ROIs
        self.rois2idx = {'None': -1,
                         'IFGorb': 0,
                         'IFGtri': 1,
                         'TP': 2,
                         'TPJ': 3,
                         'aSTS': 4,
                         'pSTS': 5
        }
        self.idx2rois = ['IFGorb',
                         'IFGtri',
                         'TP',
                         'TPJ',
                         'aSTS',
                         'pSTS',
                         None]

    def add_roi(self, roi_name):
        if roi_name not in self.rois2idx.keys():
            self.rois2idx[roi_name] = len(self.idx2rois) -1
            self.idx2rois = self.idx2rois[:-1] + [roi_name, None]

