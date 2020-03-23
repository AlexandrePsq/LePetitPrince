

class Task(object):
    """ General framework regrouping the different tasks
    possible to integrate in the pipeline.
    """
    
    def __init__(self, functions, parents=[], name):
        self.parents = parents
        self.children = None
        self.terminated = False
        self.functions = functions
        self.name = name
        pass
    
    def set_children(self, children):
        self.children = children
        
    def execute(self, *kwargs):
        if self.functions:
            if not self.is_waiting():
                func = self.function.pop()
                func(*kwargs)
                self.terminated = True
            else:
                print('Depencies not terminated...')
    
    def get_dependencies(self):
        return self.parents
        
    def get_children(self):
        return self.children

    def is_waiting(self):
        result = True
        for parent in self.parents:
            result = result and parent.terminated
        return (not result)

    def is_terminated(self):
        return self.terminated
