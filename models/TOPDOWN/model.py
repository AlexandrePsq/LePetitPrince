################################################################
# General Language Model Topdown Parser
#
################################################################
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.append(root)


import pandas as pd
import numpy as np
from utilities.settings import Params, Paths

params = Params()
paths = Paths()

class Topdown(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, language):
        super(Topdown, self).__init__()
        self.language = language


    def __name__(self):
        return 'topdown'


    def generate(self, path, language, textgrid):
        # create specific onsets-offsets
        source = 'text'
        model_category = 'TOPDOWN'
        name = os.path.basename(os.path.splitext(path)[0])
        run_name = name.split('_')[-1] # extract the name of the run
        run_nb = run_name[-1]
        data = pd.read_csv(os.path.join(paths.path2data, 'text', language, 'TOPDOWN', '{}_topdown.csv'.format(run_nb))).dropna(axis=0)
        df = pd.DataFrame({})
        df['offsets'] = data['onset'].values # the formerly saved onset is in fact the offset...
        df['onsets'] = np.nan
        df['duration'] = 0
        saving_in = '{}_{}_{}_{}_{}.csv'.format(source, language, model_category, 'onsets-offsets', run_name)
        df.to_csv(os.path.join(paths.path2data, source, language, model_category, 'onsets-offsets', saving_in), index=False)
        return pd.DataFrame(data['amplitude'].values, columns=['topdown'])
