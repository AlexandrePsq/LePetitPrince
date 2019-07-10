import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utilities.settings import Params, Paths
import pickle

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################


