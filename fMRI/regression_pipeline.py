import os


class Pipeline(object):
    """ General class to pipe the different steps of the encoding analysis.
    Allows flexible result aggregation between the functions of the defined
    flow.
    """
    
    def __init__(self):
        pass
    
    def fit(self, steps):
        self.steps = steps
    
    def aggregate(self):
        pass
    
    def distribute(self):
        pass
    
    def map(self):
        pass

    def compute(self, deep_representations, fmri_data, output_path, logs):
        pass
