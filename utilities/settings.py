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
from itertools import combinations
import numpy as np
import torch

class Paths:
    def __init__(self):
        # Paths
        self.path2root = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince'
        # self.path2root = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince'
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
                           'design-matrices':'.csv'
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
                             'run9':368,
        },
                       'french':{'run1':309,
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
        self.rois2idx = {'All': -1,
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
                         'All']

    def add_roi(self, roi_name):
        if roi_name not in self.rois2idx.keys():
            self.rois2idx[roi_name] = len(self.idx2rois) -1
            self.idx2rois = self.idx2rois[:-1] + [roi_name, 'All']


class Preferences:
	def __init__(self):

		# Number of voxel
		self.subset = None

        # LSTM
		self.seed = 1111
		self.eval_batch_size = 20
		self.bsz = 20
		self.bptt = 35 # sequence length
		self.clip = 0.25 # gradient clipping
		self.log_interval = 200 # report interval
		self.lr = 10 # learning rate
		self.epochs = 30

		# activations
		self.extracted_parameters = ['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']
		self.surprisal = False
		
		# Crossvalidation preferences
		self.ridge_nested_crossval = True
		self.defaut_alpha = 15
		self.n_alphas = 30

		# Alpha for nested
		self.alphas = np.logspace(-3, -1, 30) # Alphas list for voxel-wised analysis
		self.alpha_order_min = -4
		self.alpha_order_max = 6
		self.alphas_nested_ridgecv = np.logspace(self.alpha_order_min, self.alpha_order_max, self.n_alphas)
		self.fit_intercept = True
		
		# Data
		self.generate_data = True
		self.compute_PCA = True
		self.n_components = 4

        # GLM / Ridge
		self.detrend = True
		self.standardize = True


class Params:
	def __init__(self):
		self.pref = Preferences()

		# Data
		self.tr = 2 # FMRI sampling period
		self.nb_runs = 9 # number of runs
		self.models = sorted(['wordrate_model',
								'lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01_hidden',
								'lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01_hidden-surprisal',
								'lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01_hidden_first-layer',
								'lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01_hidden_second-layer',
								'lstm_wikikristina_embedding-size_200_nhid_100_nlayers_3_dropout_01_hidden_third-layer',
								'lstm_wikikristina_embedding-size_200_nhid_150_nlayers_2_dropout_01_hidden',
								'lstm_wikikristina_embedding-size_200_nhid_150_nlayers_2_dropout_01_hidden_first-layer',
								'lstm_wikikristina_embedding-size_200_nhid_150_nlayers_2_dropout_01_hidden_second-layer',
								'lstm_wikikristina_embedding-size_200_nhid_300_nlayers_1_dropout_01_hidden'])
		self.aggregated_models = self.models
        # self.aggregated_models = ['+'.join(item) for i in range(1, len(self.models)+1) for item in combinations(self.models, i)] ## Aggregated models (for design matrices contruction)
		self.languages = ['english'] # ['english', 'french', 'chineese']

		self.test = True
		self.overwrite = False
		self.parallel = True
		self.cuda = True
		if torch.cuda.is_available():
			if not self.cuda:
				print("WARNING: You have a CUDA device, so you should probably run with --cuda")

		self.eos_separator = '<eos>'
		self.seed = 1111

		self.nb_features_lstm = 1300
		self.features_of_interest = list(range(1301)) + [1601, 1602, 1603, 1604, 1605] # + list(range(100, 120))))

		# PCA
		self.n_components = 100

		# Scaling
		self.scaling_mean = True
		self.sclaing_var = False
		
	def get_category(self, model_name):
		category = model_name.split('_')[0]
		return category.upper()