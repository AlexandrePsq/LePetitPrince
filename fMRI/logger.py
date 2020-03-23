from utils import write


class Logger(object):
    """ Framework to log encoding analysis results.
    """
    
    def __init__(self, path):
        self.log_path = path
    
    def report_logs(self, logs):
        write(self.log_path, logs)
