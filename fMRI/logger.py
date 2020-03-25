from utils import write


class Logger(object):
    """ Framework to log encoding analysis results.
    """
    
    def __init__(self, path):
        self.log_path = path
    
    def report_logs(self, logs, level):
        write(self.log_path, level, end=': ')
        write(self.log_path, logs)
    
    def error(self, message):
        self.report_logs(self.log_path, message, level='ERROR')
        raise Exception(message)
    
    def warning(self, message):
        self.report_logs(self.log_path, message, level='WARNING')

    def info(self, message):
        self.report_logs(self.log_path, message, level='INFO')
