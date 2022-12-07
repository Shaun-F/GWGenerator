import numpy as np
from .HelperFunctions import *

class Evolution():

    def __init__(self, InstantiatedKerrClass, InstantiatedProcaClass):
        self.Kerr = InstantiatedKerrClass
        self.PC = InstantiatedProcaClass

    def FinalFrequency(self,m,M0,a0,w0, Threshold = 10**(-3)):
        """deprecated"""
        return NotImplementedError

    def FinalSpin(self,m,M0,a0,w):
        Mf = self.FinalMass(m,M0,a0,w)
        Mbar = Mf/M0
        res = (m/w)*(1/(Mbar*M0))*(1-1/Mbar) + a0*1/(Mbar**2)
        return res

    def FinalMass(self,m,M0,a0,w):
        res = m**3/(8.*M0**2*w**2*(m - a0*M0*w)) - Sqrt(m**6 - 16*m**4*M0**2*w**2 + 32*a0*m**3*M0**3*w**3 - 16*a0**2*m**2*M0**4*w**4)/(8.*M0**2*w**2*((m - a0*M0*w)))
        return res*M0
