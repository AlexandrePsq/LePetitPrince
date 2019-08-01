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
from itertools import product
import numpy as np
import torch

class Paths:
    def __init__(self):
        # Paths
        self.path2root = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince'
        #self.path2root = '/Users/alexpsq/Code/NeuroSpin/LePetitPrince'
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
		self.epochs = 40
		self.shift_surprisal = 0

		# activations
		self.extracted_parameters = ['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']
		self.surprisal = True
		self.entropy = True
		
		# Crossvalidation preferences
		self.ridge_nested_crossval = True
		self.n_alphas = 30

		# Alpha for nested
		self.alphas = np.logspace(-3, 3, 30) # Alphas list for voxel-wised analysis
		self.alpha_order_min = -4
		self.alpha_order_max = 6
		self.alphas_nested_ridgecv = np.logspace(self.alpha_order_min, self.alpha_order_max, self.n_alphas)
		self.fit_intercept = True

        # GLM / Ridge
		self.detrend = True
		self.standardize = True
		self.alpha_default = 100


class Params:
	def __init__(self):
		self.pref = Preferences()

		# Data
		self.tr = 2 # FMRI sampling period
		self.nb_runs = 9 # number of runs
		self.models = sorted(['lstm_wikikristina_embedding-size_600_nhid_1800_nlayers_1_dropout_02_hidden_first-layer'])
		#self.models = sorted(['lstm_wikikristina_embedding-size_600_nhid_50_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_100_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_150_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_200_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_300_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_400_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_500_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_600_nlayers_1_dropout_02_hidden_first-layer',
		#						'lstm_wikikristina_embedding-size_600_nhid_1200_nlayers_1_dropout_02_hidden_first-layer',
		# 						'lstm_wikikristina_embedding-size_600_nhid_1800_nlayers_1_dropout_02_hidden_first-layer'])
		self.aggregated_models = self.models
		self.basic_features = sorted(['wordrate_model', 'rms_model']) #, 'word_freq',  'fundamental_freq'
		self.modelsOfInterest = sorted(['lstm_wikikristina_embedding-size_600_nhid_600_nlayers_1_dropout_02_hidden',
										'lstm_wikikristina_embedding-size_600_nhid_600_nlayers_2_dropout_02_hidden',
										'lstm_wikikristina_embedding-size_600_nhid_600_nlayers_3_dropout_02_hidden',
										'lstm_wikikristina_embedding-size_600_nhid_200_nlayers_3_dropout_02_hidden'])
		#self.aggregated_models = ['+'.join(model) for model in list(product(['+'.join(self.basic_features)], self.modelsOfInterest))]
		# self.aggregated_models = ['+'.join(item) for i in range(1, len(self.models)+1) for item in combinations(self.models, i)] ## Aggregated models (for design matrices contruction)
		self.languages = ['english'] # ['english', 'french', 'chineese']

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
		self.test = True
		self.overwrite = False
		self.parallel = True
		self.cuda = True
		self.voxel_wise = False
		if torch.cuda.is_available():
			if not self.cuda:
				print("WARNING: You have a CUDA device, so you should probably run with --cuda")

		self.eos_separator = '<eos>'
		self.seed = 1111

		self.nb_features_lstm = 1300
		self.features_of_interest = list(range(1301)) + [1601, 1602, 1603, 1604, 1605] # + list(range(100, 120))))

		# PCA
		self.pca = False
		self.n_components_default = 1500
		self.n_components_list = [500, 800, 1500] #[1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500, 1800]

		# Deconfounding
		self.alpha_percentile = 99

		# Scaling
		self.scaling_mean = True
		self.scaling_var = False
		
	def get_category(self, model_name):
		category = model_name.split('_')[0]
		return category.upper()