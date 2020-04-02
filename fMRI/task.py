"""
General framework regrouping the different tasks possible to integrate in the 
regression analysis pipeline.
===================================================
A Task instanciation requires:
    - functions: a list of functions to be executed sequentially on each input,
    - input_dependencies: list of Tasks whose outputs are to be used for this 
    particular task,
    - name: name (string) of the task,
    - flatten_inputs: list of bool specifying if we have to flatten the outputs of 
    the tasks in iputs_dependencies, 
    - unflatten_output: ('automatic' / int / None), specifying if we unflatten the output of this
    particular task, 
    - special_output_transform: function that should be applied on the final output
    of the task.
The task is executed through the method self.execute() which aggregates the output of 
parent tasks to give it as input to the current task.
It then apply sequentially the functions in self.functions on each item of its input.
"""

from utils import merge_dict, filter_args, save
from tqdm import tqdm



class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, functions=None, input_dependencies=[], name='', flatten_inputs=None, unflatten_output=None, special_output_transform=None):
        """ Instanciation of a task.
        Arguments:
            - functions: list (of functions)
            - input_dependencies: list (of Tasks)
            - name: str
            - flatten_inputs: list (of bool)
            - unflatten_output: 'automatic' / int / None
            - special_output_transform: function
        """
        self.input_dependencies = input_dependencies
        self.children = []
        self.terminated = False
        self.functions = functions
        self.name = name
        self.output = []
        if flatten_inputs and (len(flatten_inputs)==len(input_dependencies)):
            self.flatten = flatten_inputs if flatten_inputs else [False for item in input_dependencies]
        else:
            self.flatten = [False for item in input_dependencies]
        self.unflatten = unflatten_output
        self.unflatten_factor = unflatten_output if isinstance(unflatten_output, int) else None
        self.special_output_transform= special_output_transform
    
    def set_children_tasks(self, children):
        """ Set self.children value."""
        self.children = children
    
    def set_terminated(self, bool_value):
        """ Set self.terminated value."""
        self.terminated = bool_value
    
    def set_output(self, value):
        """ Set self.output value."""
        self.output = value

    def update_flatten(self):
        """Update self.flatten value."""
        if len(self.input_dependencies) > 0:
            if (not self.flatten) or (len(self.input_dependencies) != len(self.flatten)):
                self.flatten = [False for item in self.input_dependencies]
    
    def add_input_dependencies(self, parent):
        """Add parent task to current task.
        Arguments:
            - parent: Task
        """
        self.input_dependencies.append(parent)
        self.update_flatten()
    
    def add_output(self, output):
        """ Add value to self.output."""
        self.output.append(output)
    
    def get_input_dependencies(self):
        """ Get parent tasks."""
        return self.input_dependencies
        
    def get_children(self):
        """ Get children tasks."""
        return self.children

    def is_waiting(self):
        """ Check if the task is temrinated."""
        result = True
        for parent in self.input_dependencies:
            result = result and parent.is_terminated()
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
            save(result, path + '_' + str(index))
    
    def flatten_(self, input_, index):
        """ Flatten a given input.
        Arguments:
            - input_: list (of list)
            - index: int
        """
        if self.flatten[index]:
            if self.unflatten=='automatic':
                self.unflatten_factor = len(input_[0])
            input_ = [item for sublist in input_ for item in sublist]
        return input_
    
    def unflatten_(self):
        """ Unflatten the output of the task when we have flattened
        the input.
        """
        if self.unflatten:
            self.output = [self.output[x : x + self.unflatten_factor] for x in range(0, len(self.output), self.unflatten_factor)]
    
    def execute(self):
        """ Execute all task functions on the serie of parents outputs."""
        if not (self.is_waiting() or self.is_terminated()):
            inputs_ =  list(zip(*[self.flatten_(parent.output, index) for index, parent in enumerate(self.input_dependencies)])) # regroup dictionaries outputs from parent tasks
            inputs_ = [list(item) for item in inputs_] # transform tuple to list -> problematic when 1 single parent
            inputs = [merge_dict(items) for items in inputs_]
            for input_ in tqdm(inputs):
                input_tmp = input_.copy()
                for func in self.functions:
                    input_tmp = filter_args(func, input_tmp)
                    input_tmp = func(**input_tmp)
                self.add_output(input_tmp)
            self.set_terminated(True)
            self.unflatten_()
            if self.special_output_transform:
                self.output = self.special_output_transform(self.output)
        else:
            print('Dependencies not fullfilled...')
    