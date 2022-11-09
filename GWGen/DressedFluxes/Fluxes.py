import numpy as np
import pandas as pd
import scipy as sp
import os
import superrad as SR

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'/../../ProcaSolutions/';

class ProcaSolution():
    def __init__(self, BHMass, BHSpin, ProcaMass, units="natural"):
        self.BosonCloud = SR.ultralight_boson.UltralightBoson(spin=1, model="relativistic")
        self.BosonWaveform = self.BosonCloud.make_waveform(BHMass, BHSpin, ProcaMass, units=units)

    def GWFlux(self,t=0)
        #Convert units!!!! 
        return self.BosonWaveform.power_gw(t)

    def GWTimescale(self):
        return self.BosonWaveform.gw_time()




class OldProcaSolution():
    def __init__(self):
        self.SolutionDir = pathToSolutionSet;
        self.SolutionFile = self.SolutionDir+'SolutionSet.dat';
        self.PDDataFrame=pd.read_csv(self.SolutionFile, delimiter='\t')

    def GenerateFluxInterpolation(self, modenumber=1, overtone=0, BHspin=0.9):
        restricteddataset = self.PDDataFrame[(self.PDDataFrame["ModeNumber"]==modenumber)&(self.PDDataFrame["Overtone"]==overtone)&(self.PDDataFrame["BlackHoleSpin"]==BHspin)]
        ProcaMasses = restricteddataset["ProcaMass"].values.ravel()
        Einf = restricteddataset["Einf"].values.ravel()
        interpolatingfunction = sp.interpolate.CubicSpline(ProcaMasses, Einf)
        res = lambda x: float(interpolatingfunction(x))
        return res

    def ProcaFlux(self, ProcaMass=0.3, ModeNumber=1, OvertoneNumber=0, BHSpin=0.9):
        interp = self.GenerateFluxInterpolation(modenumber=ModeNumber,overtone=OvertoneNumber, BHspin=BHSpin)
        return interp(ProcaMass)
