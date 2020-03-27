
from task import Task



class Pipeline(object):
    """ General class to pipe the different steps of the encoding analysis.
    Allows flexible result aggregation between the functions of the defined
    flow.
    """
    
    def __init__(self):
        pass
    
    def reset_tasks(self):
        """ Reset all tasks in the pipeline."""
        for task in self.tasks:
            task.set_terminated(False)
    
    def fit(self, task, logger):
        """ Determine in which order to run the tasks depending on their dependencies.
        Arguments:
            - task: Task
            - logger: Logger object
        """
        self.tasks = []
        queue = [task]
        count = 1
        while queue:
            task = queue.pop()
            if task.is_waiting():
                if count==0:
                    logger.error("Loop detected... Please check tasks dependencies.")
                else:
                    queue = [task] + queue
                    count -= 1
            else:
                self.tasks.append(task)
                task.set_terminated(True)
                queue = task.children + queue
                count += len(task.children)
        self.reset_tasks()
        logger.info("The pipeline was fitted without error.")
            
    def compute(self, X_train, Y_train, output_path, logger):
        """ Execute pipeline.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - output_path: str
            - logger: Logger object
        """
        inputs = [{'X_train':X_train, 'Y_train':Y_train}]
        if not self.tasks:
            logger.warning("Pipeline not fitted... Nothing to compute.")
        else:
            empty_task = Task()
            empty_task.set_output(inputs)
            empty_task.set_terminated(True)
            self.tasks[0].dependencies = [empty_task]
            for index, task in enumerate(self.tasks):
                logger.info("{}. Executing task: {}".format(index, task.name))
                task.execute()
                logger.validate()
            logger.info("Saving output...")
            task.save_output(output_path)
            logger.validate()
        logger.info("The pipeline was executed without error.")
        logger.validate()
        return task.output
