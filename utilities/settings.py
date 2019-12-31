from os.path import join
from itertools import product
import numpy as np
import torch

class Paths:
    def __init__(self):
        # Paths
        self.path2root = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince'
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
                           'raw-features':'.csv',
                           'features':'.csv',
                           'fMRI':'.nii.nii',
                           'design-matrices':'.csv',
						   'predictions': '.csv'
        }

    def get_extension(self, data_type):
        return self.extensions[data_type]
        
class Scans:
    def __init__(self):
        # number of scans per run
        self.nscans = {'english':{'run1':282,
                             'run2':298,
                             'run3':340,
                             'run4':303,
                             'run5':265,
                             'run6':343,
                             'run7':325,
                             'run8':292,
                             'run9':368
        },
                       'french':{'run1':309,
                             'run2':326,
                             'run3':354,
                             'run4':315,
                             'run5':293,
                             'run6':378,
                             'run7':332,
                             'run8':294,
                             'run9':336
        }
        }
    def get_nscans(self, language, run):
                       return self.nscans[language][run]
        
class Subjects:
    def __init__(self):
        # Subjects
        self.subject_lists = {'english': [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93, 
        94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115],
                              'french': []
        }
        self.subject_test = {'english': [57],
                             'french': []
        }
        
    def get_subject(self, subject_number):
        if subject_number < 10:
            return 'sub-00{}'.format(subject_number)
        elif subject_number < 100:
            return 'sub-0{}'.format(subject_number)
        else:
            return 'sub-{}'.format(subject_number)

    def get_all(self, language):
        result = []
        sub_list = self.subject_lists[language]
        for subj in sub_list:
            result.append(self.get_subject(subj))
        return result


class Preferences:
	def __init__(self):

        # LSTM
		self.eval_batch_size = 20
		self.bsz = 20
		self.bptt = 35 # sequence length
		self.clip = 0.25 # gradient clipping
		self.log_interval = 200 # report interval
		self.lr = 10 # learning rate
		self.epochs = 40
		self.shift_surprisal = 0

		# activations
		self.extracted_parameters = ['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']
		self.surprisal = True
		self.entropy = True
		
		# Crossvalidation preferences
		self.ridge_nested_crossval = True
		self.n_alphas = 25

		# Alpha for nested
		self.alphas = np.logspace(-3, 3, 30) # Alphas list for voxel-wised analysis
		self.fit_intercept = True

        # GLM / Ridge
		self.detrend = True
		self.standardize = True
		self.alpha_default = 100


class Params:
	def __init__(self):
		self.pref = Preferences()

		# ROI
		self.atlas = 'cort-prob-2mm' #extracted from harvard-oxford
		self.atlas_possible = ['cort-maxprob-thr0-1mm',
								'cort-maxprob-thr0-2mm',
								'cort-maxprob-thr25-1mm',
								'cort-maxprob-thr25-2mm',
								'cort-maxprob-thr50-1mm',
								'cort-maxprob-thr50-2mm',
								'cort-prob-1mm',
								'cort-prob-2mm',
								'cortl-maxprob-thr0-1mm',
								'cortl-maxprob-thr0-2mm',
								'cortl-maxprob-thr25-1mm',
								'cortl-maxprob-thr25-2mm',
								'cortl-maxprob-thr50-1mm',
								'cortl-maxprob-thr50-2mm',
								'cortl-prob-1mm',
								'cortl-prob-2mm']


		# general parameters
		self.eos_separator = '<eos>'