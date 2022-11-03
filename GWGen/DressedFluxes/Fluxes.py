import numpy as np
import os

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'../../ProcaSolutions';

class ProcaSolution():
    def __init__(self):
        self.SolutionDir = pathToSolutionSet;
        self.SolutionFile = self.SolutionDir+'SolutionSet.dat';

    def importdata(self):
        
