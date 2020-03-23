import os


class Transformer(object):
    """ Perform general transformations over a set of dataframes or numpy arrays,
    taking into account group structures.
    """
    
    
    def __init__(self):
        pass
    
    def standardize(self):
        pass
    
    def scale(self):
        pass
        
    def make_regressor(self):
        pass
    
    def process_representations(self, representation_paths, models):
        """ Load representation dataframes and create the design matrix
        for each run.
        """
        runs = list(zip(*representation_paths)) # list of 9 tuples (1 for each run), each tuple containing the representations for the specified models
        # e.g.: [(path2run1_model1, path2run1_model2), (path2run2_model1, path2run2_model2)]

        # Computing design-matrices
        for i in range(len(runs)):
            merge = pd.concat([pd.read_csv(path2features, header=0)[eval(models[index]['columns2retrieve'])] for index, path2features in enumerate(runs[i])], axis=1) # concatenate horizontaly the read csv files of a run
            dataframes.append(merge)
        return dataframes
    
    def process_fmri_data(self, fmri_paths, masker):
        """ Load fMRI data and mask it with a given masker.
        Preprocess it to avoid NaN value when using Pearson
        Correlation coefficients in the following analysis.
        """
        data = [masker.transform(f) for f in fmri_paths]
        # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
        for run in range(len(data)):
            zero = np.zeros(data[run].shape[0])
            new = zero
            new[0] += np.random.random()/1000
            data[run] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, data[run])
        return data
