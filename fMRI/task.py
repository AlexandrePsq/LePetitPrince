

class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, function, parents=[]):
        self.parents = parents
        self.terminated = False
        self.function = function
        pass
        
    def execute(self, **kwargs):
        execute_ = True
        for parent in self.parents:
            execute_ = execute_ and parent.terminated
        if execute_:
            self.function(**kwargs)
            self.terminated = True
        else:
            print('Depencies not terminated...')
    
    def get_dependencies(self):
        return self.parents
