import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate as spint
import os
import superrad as SR
from superrad import ultralight_boson
import astropy.constants as cons
import astropy.units as unit
from GWGen.Utils import *
import re

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'/../../Analytic_Flux/ProcaEnDenCSVData/';

class ProcaSolution():
	def __init__(self, BHMass, BHSpin, ProcaMass, BosonSpin=1,CloudModel = "relativistic",units="physical"):
			if units=="natural":
				raise NotImplementedError("natural units not yet implemented. Use physical units. See https://bitbucket.org/weast/superrad/src/master/ for definitions")
			self.SMBHMass = BHMass
			self.SMBHSpin = BHSpin
			self.ProcaMass = ProcaMass
			self.units = units
			self.BosonSpin = BosonSpin
			self.CloudModel = CloudModel
			self.BosonCloud = SR.ultralight_boson.UltralightBoson(spin=self.BosonSpin, model=self.CloudModel)

			try:
				self.BosonWaveform = self.BosonCloud.make_waveform(BHMass, BHSpin, ProcaMass, units=units)
			except ValueError as err:
				print("Error in Proca Solution: \n {0}".format(err))

			self.Kerr = Kerr(BHSpin=BHSpin)
			self.alpha = alphavalue(self.SMBHMass, self.ProcaMass)
			self.enden = self.GetEnergyDensity()


	def ChangeInOrbitalEnergy(self):
		fractionalenden = self.FractionalGWEFlux() #anonymous function in (t,p)
		deltaEdeltaM = self.Kerr.dEdM() #anonymous function in (e,p)
		DeltaOrbitalEnergy = lambda t,e,p: deltaEdeltaM(e,p)*fractionalenden(t,p)

		return DeltaOrbitalEnergy

	def FractionalGWEFlux(self):
		FractionalFactor = lambda p:self.FractionalEnergyDensity(p)**2
		FractionalEnDen = lambda t,p: FractionalFactor(p)*self.BosonCloudGWEFlux(t)

		return FractionalEnDen


	def FractionalEnergyDensity(self, r):
		rstart = self.coorddata[0][0]
		rmax = self.coorddata[0].dropna().values[-1]
		thstart = self.coorddata[1][0]
		thstop = self.coorddata[1].dropna().values[-1]
		toten = lambda r: self.enden.integral(rstart, r, thstart, thstop)
		val = toten(r)/toten(rmax)
		return val

	def GetEnergyDensity(self):
		coordvals = self._get_closest_alpha_filename(self.alpha)
		enden = self._generate_interp(coordvals)

		return enden

	def _get_closest_alpha_filename(self, alpha):

		#extract alpha value from filename
		allfilenames = os.listdir(pathToSolutionSet)
		alpha_regex = re.compile('Alpha_\d+_\d+')
		alpha_matches = [alpha_regex.findall(string) for string in allfilenames]
		value_regex = re.compile('\d+')
		numdoms = [list(map(float, value_regex.findall(st[0]))) for st in alpha_matches]
		alphas = [i[0]/i[1] for i in numdoms]

		#find closest alpha value to input parameter
		tmp = np.abs(alpha-alphas)
		indicies = np.where(tmp==tmp.min())
		result = np.array(allfilenames)[indicies]

		#get coordinate and value data
		sorting_inx = [st[13:19]=='COORDS' for st in result]
		coordret = result[sorting_inx][0]
		valueret = np.setdiff1d(result, coordret)[0]

		#return dictionary
		return_value = {"COORDS":coordret, "VALUES":valueret}

		return return_value

	def _generate_interp(self, coordvaluefilenamedict):
		self.coorddata = pd.read_csv(pathToSolutionSet+coordvaluefilenamedict["COORDS"], header=None, na_values="None")
		self.valuedata = pd.read_csv(pathToSolutionSet+coordvaluefilenamedict["VALUES"], header=None, na_values="None")

		radial_data = self.coorddata[0].dropna()
		theta_data = self.coorddata[1].dropna()


		interp = spint.RectBivariateSpline(radial_data, theta_data, self.valuedata)
		return interp


	def BosonCloudGWEFlux(self,t=0):
			#Must divide by mass ratio m/M to get actual dimensionless power, where m is the mass of the secondary
			## See definition of dimenionless energy and dimensionless time
			DimensionfullPower = self.BosonWaveform.power_gw(t)*unit.watt
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






"""
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
"""
