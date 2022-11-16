import numpy as np
import pandas as pd
import scipy as sp
import os
import superrad as SR
from superrad import ultralight_boson
import astropy.constants as cons
import astropy.units as unit

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'/../../ProcaSolutions/';

class ProcaSolution():
	def __init__(self, BHMass, BHSpin, ProcaMass, BosonSpin=1,CloudModel = "relativistic",units="physical"):
			if units=="natural":
				raise NotImplementedError("natural units not yet implemented. Use physical units. See https://bitbucket.org/weast/superrad/src/master/ for definitions")
			self.SMBHMass = BHMass
			self.SMBHSpin = BHSpin
			self.units = units
			self.BosonSpin = BosonSpin
			self.CloudModel = CloudModel
			self.BosonCloud = SR.ultralight_boson.UltralightBoson(spin=self.BosonSpin, model=self.CloudModel)
			self.BosonWaveform = self.BosonCloud.make_waveform(BHMass, BHSpin, ProcaMass, units=units)

	def BosonCloudGWEFlux(self,t=0):
			#Must divide by mass ratio m/M to get actual dimensionless power, where m is the mass of the secondary
			## See definition of dimenionless energy and dimensionless time
			DimensionfullPower = self.BosonWaveform.power_gw(t)*unit.watt
			MassOfCloud = self.BosonCloudMass(t)*unit.Msun #convert energy of cloud in units of cloud mass
			MassOfSMBH = self.SMBHMass * unit.Msun #express time in units of SMBH mass
			conversion = (MassOfSMBH/MassOfCloud)*(cons.G/(cons.c**5))
			res = (DimensionfullPower).decompose() # dimensionfull power
			return -res
	def BosonCloudGWLFlux(self,t=0):
			Eflux = self.BosonCloudGWEFlux(t) #dimensionless power
			azimuthalnumber = self.BosonWaveform.azimuthal_num()
			frequency = self.BosonWaveform.freq_gw(t)*unit.Hz
			dimlessfrequency = (cons.G*self.SMBHMass*unit.Msun/(cons.c**3))*frequency*unit.Hz #in units of SMBH frequency
			return (azimuthalnumber*Eflux/frequency).decompose() #all power emitted in single proca mode. In units of cloud energy Mc * c^2

	def BosonCloudGWTimescale(self):
        	return self.BosonWaveform.gw_time()

	def BosonCloudMass(self,t=0):
			return self.BosonWaveform.mass_cloud(t)



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
