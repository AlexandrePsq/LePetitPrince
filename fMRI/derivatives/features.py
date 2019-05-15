import argparse
from os.path import join


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ..utilities.settings import Paths, Scans
from ..utilities.utils import get_data
import pandas as pd
import numpy as np
from nistats.hemodynamic_models import compute_regressor


paths = Paths()
scans = Scans()


def process_raw_features(run, tr, nscans):
    df = pd.read_csv(run)
    conditions = np.vstack((df.onset, df.duration, df.amplitude))
    result = compute_regressor(exp_condition=conditions,
                               hrf_model='spm',
                               frame_times=np.arange(0.0, nscans*tr, tr)
                               oversampling=10)
    return pd.DataFrame(result[0], columns=['hrf'])



if __name =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate features (=fMRI regressors) from raw_features (csv file with 3 columns onset-amplitude-duration) by convolution with an hrf kernel.\n\nInput:\nTime resolution of the scans, language and models.""")
    parser.add_argument("--tr", type=int, default=None, help="TR in sec (=sampling period for scans)")
    parser.add_argument("--models", nargs="+", action="append", default=[], help="Names of the models used to generate the raw features")
    parser.add_argument("--language", type=str, default='en', help="Language of the models studied.")
    parser.add_argument("--test", type=bool, default=False, action="store_true", help="Precise if we are running a test.")

    args = parser.parse_args()

    for model_ in args.models[0]:

        data_type = 'raw_features'
        data = get_data(args.language, data_type, model_, source='fMRI', args.test)
            
        for i, run in enumerate(data):
            name = os.path.basename(run)
            run_name = name.split('_')[-1]
            path2output = join(paths.path2derivatives, "features/{0}/{1}/features_{0}_{1}_{2}".format(args.language, model_, run_name))
            nscans = scans.get_nscans(run_name)
            df = process_raw_features(run, agrs.tr, nscans)
            df.to_csv(path2output, index=False, header=False)
