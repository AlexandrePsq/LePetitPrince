from utils import merge_dict, filter_args



class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, functions=None, parents=[], name=''):
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
    
    def execute(self):
    """ Execute all task functions on the serie of parents outputs."""
    if not (self.is_waiting() or self.is_terminated()):
        inputs_ =  list(zip(*[parent.output for parent in self.parents]))
        inputs = [merge_dict(list(items)) for items in inputs_]
        for input in inputs:
            input_tmp = input.copy()
            for func in self.functions:
                input_tmp = filter_args(func, input_tmp)
                input_tmp = func(**input_tmp)
            self.add_output(input_tmp)
        self.terminated = True
    else:
        print('Depencies not terminated...')
    
    
