import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler



class Transformer(object):
    """ Perform general transformations over a set of dataframes or numpy arrays,
    taking into account group structures.
    """
    
    
    def __init__(self, tr, nscans, hrf='spm', oversampling=10):
        self.tr = tr
        self.nscans = nscans
        self.hrf = hrf
        self.oversampling = oversampling
    
    def standardize(self, matrices):
        """Standardize a list of matrices.
        Arguments:
            - matrices: list (of np.array)
        """
        matrices = matrices if isinstance(matrices, list) else [matrices]
        for index in range(len(matrices)):
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(matrices[index])
            matrices[index] = scaler.transform(matrices[index])
        return matrices
    
    def make_regressor(self, matrix):
        """ Compute the convolution with an hrf for each column of the matrix.
        Arguments:
            - array: Pandas.Dataframe
            - tr: float
            - nscans: int
        """
        regressors = []
        representations = [col for col in matrix.columns if col not in ['offsets', 'duration']]
        for col in representations:
            conditions = np.vstack((matrix.offsets, matrix.duration, matrix[col]))
            tmp = compute_regressor(exp_condition=conditions,
                                    hrf_model=self.hrf,
                                    frame_times=np.arange(0.0, self.nscans * self.tr, self.tr),
                                    oversampling=self.oversampling)
            regressors.append(pd.DataFrame(tmp[0], columns=[col]))
        result = pd.concat(regressors, axis=1)
        return result
    
    def process_representations(self, representation_paths, models):
        """ Load representation dataframes and create the design matrix
        for each run.
        Arguments:
            - representation_paths: list (of list of paths)
            - models: list (of dict)
        """
        arrays = []
        runs = list(zip(*representation_paths)) # list of 9 tuples (1 for each run), each tuple containing the representations for the specified models
        # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]

        # Computing design-matrices
        for i in range(len(runs)):
            merge = pd.concat([pd.read_csv(path2features, header=0)[eval(models[index]['columns2retrieve'])] for index, path2features in enumerate(runs[i])], axis=1) # concatenate horizontaly the read csv files of a run
            arrays.append(merge.data)
        return arrays
    
    def process_fmri_data(self, fmri_paths, masker):
        """ Load fMRI data and mask it with a given masker.
        Preprocess it to avoid NaN value when using Pearson
        Correlation coefficients in the following analysis.
        Arguments:
            - fmri_paths: list (of string)
            - masker: NiftiMasker object
        """
        data = [masker.transform(f) for f in fmri_paths]
        # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
        for run in range(len(data)):
            zero = np.zeros(data[run].shape[0])
            new = zero
            new[0] += np.random.random()/1000
            data[run] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, data[run])
        return data
