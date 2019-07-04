import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utilities.settings import Params, Paths

params = Params()
paths = Paths()

###############################################################################
# Utilities
###############################################################################



#def save(model, data_name, language):
#    path = '_'.join([model.__name__(), data_name, language]) + '.pt'
#    path = os.path.join(paths.path2derivatives, 'fMRI/models', language, path)
#    with open(path, 'wb') as f:
#        torch.save(model, f)
#
#
#def load(model, data_name, language):
#    path = '_'.join([model.__name__(), data_name, language]) + '.pt'
#    path = os.path.join(paths.path2derivatives, 'fMRI/models', language, path)
#    assert os.path.exists(path)
#    with open(path, 'rb') as f:
#        return torch.load(f)