import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)


from utilities.settings import Subjects, Rois, Paths, Params
from utilities.utils import get_output_parent_folder, get_path2output
from itertools import combinations, product
import numpy as np

import warnings
warnings.simplefilter(action='ignore')


from os.path import join

############################################################################
################################# Analysis #################################
############################################################################


#####################################################################
############ Model comparison with equivalent perplexity ############
#####################################################################
params = Params()