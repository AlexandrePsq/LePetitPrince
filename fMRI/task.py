

class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, functions, parents=[], name):
        self.parents = parents
        self.terminated = False
        self.functions = functions
        self.name = name
        pass
        
    def execute(self, *kwargs):
        if self.functions:
            execute_ = True
            for parent in self.parents:
                execute_ = execute_ and parent.terminated
            if execute_:
                func = self.function.pop()
                func(*kwargs)
                self.terminated = True
            else:
                print('Depencies not terminated...')
    
    def get_dependencies(self):
        return self.parents
