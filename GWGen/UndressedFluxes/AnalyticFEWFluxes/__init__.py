#theres got to be a better way of importing dIdt.pyx inside the package....
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")

from .dIdt import *
