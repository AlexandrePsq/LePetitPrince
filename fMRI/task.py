from utils import merge_dict, filter_args, save



class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, functions=None, parents=[], name='', flatten=False, special_output_transform=None):
        """ Instanciation of a task.
        Arguments:
            - functions: list (of functions)
            - parents: list (of Tasks)
            - name: str
        """
        self.parents = parents
        self.children = None
        self.terminated = False
        self.functions = functions
        self.name = name
        self.output = []
        self.flatten = flatten
        self.special_output_transform= None
    
    def set_children(self, children):
        """ Set self.children value."""
        self.children = children
    
    def set_terminated(self, bool_value):
        """ Set self.terminated value."""
        self.terminated = bool_value
    
    def add_output(self, output):
        """ Add value to self.output."""
        self.output.append(output)
    
    def get_dependencies(self):
        """ Get parent tasks."""
        return self.parents
        
    def get_children(self):
        """ Get children tasks."""
        return self.children

    def is_waiting(self):
        """ Check if the task is temrinated."""
        result = True
        for parent in self.parents:
            result = result and parent.terminated
        return (not result)

    def is_terminated(self):
        """ Check if the task is temrinated."""
        return self.terminated
    
    def save_output(self, path):
        """ Save the output of the task.
        Argume,ts:
            - path: str
        """
        for index, result in enumerate(self.output):
            save(result, path + str(index))
    
    def flatten(self, input_):
        """ Flatten a given input.
        Arguments:
            - input_: list (of list)
        """
        if self.flatten:
            if not self.parents:
                raise Exception("No parents input to flatten...")
            flattening_factor = len(input_[0])
            self.flattening_factor = flattening_factor
            input_ = [item for sublist in input_ for item in sublist]
        return input_
    
    def unflatten(self):
        """ Unflatten the output of the task when we have flattened
        the input.
        """
        if self.flatten:
            self.output = [self.output[x : x + self.flattening_factor] for x in range(0, len(self.output), self.flattening_factor)]
    
    def execute(self):
    """ Execute all task functions on the serie of parents outputs."""
    if not (self.is_waiting() or self.is_terminated()):
        inputs_ =  list(zip(*[self.flatten(parent.output) for parent in self.parents]))
        inputs = [merge_dict(list(items)) for items in inputs_]
        for input in inputs:
            input_tmp = input.copy()
            for func in self.functions:
                input_tmp = filter_args(func, input_tmp)
                input_tmp = func(**input_tmp)
            self.add_output(input_tmp)
        self.terminated = True
        self.unflatten()
        if self.special_output_transform:
            self.special_output_transform(self.output)
    else:
        print('Dependencies not fullfilled...')
    
    
