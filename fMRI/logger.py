import os
from utils import write
import matplotlib.pyplot as plt
plt.switch_backend('agg')



class Logger(object):
    """ Framework to log encoding analysis progression and results.
    """
    
    def __init__(self, path):
        self.log_path = path
    
    def report_logs(self, logs, level, end):
        """General reporting function.
        Arguments:
            - logs: str
            - level: str
        """
        write(self.log_path, level, end=': ')
        write(self.log_path, logs, end=end)
    
    def error(self, message):
        """Reports ERROR messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='ERROR', end='\n')
        raise Exception(message)
    
    def warning(self, message):
        """Reports WARNING messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='WARNING', end='\n')

    def info(self, message):
        """Reports INFO messages.
        Arguments:
            - message: str
        """
        self.report_logs(message, level='INFO', end=' ')
    
    def validate(self):
        """Validate previous message."""
        write(self.log_path, '--> Done', end='\n')

    def figure(self, array):
        """Reports a figure.
        Arguments:
            - array: np.array
        """
        plt.plot(array)
        plt.savefig(os.path.join(os.path.dirname(self.log_path),'explained_variance.png'))
        plt.close()
