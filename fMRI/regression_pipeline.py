
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
        count = 0
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
            
    def compute(self, X_list, Y_list, output_path, logger):
        """ Execute pipeline.
        Arguments:
            - X_list: list (of np.array)
            - Y_list: list (of np.array)
            - output_path: str
            - logger: Logger object
        """
        inputs = [{'X_list':X_list, 'Y_list':Y_list}]
        if not self.tasks:
            logger.warning("Pipeline not fitted... Nothing to compute.")
        else:
            empty_task = Task()
            empty_task.set_output(inputs)
            self.tasks[0].parents = empty_task
            for index, task in enumerate(self.tasks):
                logger.info("{}. Executing task: {}".format(index, task.name))
                task.execute()
            logger.info("Saving output...")
            task.save_output(output_path)
        logger.info("The pipeline was executed without error.")
        return task.output
