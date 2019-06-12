import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.append(root)



from .WORDRATE import model
from .WORDRATE.utils import wordrate
import os



def load():
    # mod is only used for name retrieving ! the actual trained model is retrieved in the last line
    language = 'english'
    mod = model.Wordrate([wordrate], language)
    return mod
