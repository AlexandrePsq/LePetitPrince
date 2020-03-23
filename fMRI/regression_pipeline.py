


class Pipeline(object):
    """ General class to pipe the different steps of the encoding analysis.
    Allows flexible result aggregation between the functions of the defined
    flow.
    """
    
    def __init__(self):
        pass
    
    def fit(self, task):
        """
        """
        self.tasks = []
        queue = []
        while not task.is_terminated():
            if task.is_waiting:
                queue = task + queue # il faudra gerer le cas d'une boucle (exception)
            else:
                self.tasks.append(task)
                queue = task.children + queue
            task = queue.pop()
            
    
    def aggregate(self):
        pass
    
    def distribute(self):
        pass
    
    def map(self, func, input_list):
        """ Apply a function to the lists of arguments
        contained in input_list.
        Arguments:
            - input_list: list (of lists)
        """
        results = []
        for inputs in input_list:
            results.append(func(*inputs))
        return results

    def compute(self, deep_representations, fmri_data, output_path, logs):
        """
        """
        inputs = [(deep_representations, fmri_data)]
        if not self.tasks:
            print("Pipeline not fitted...")
        else:
            for index, task in enumerate(self.tasks):
                outputs = self.map(task.execute, inputs)
                inputs = self. # 
            pass
