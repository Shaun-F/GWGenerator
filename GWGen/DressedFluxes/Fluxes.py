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

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'/../ProcaData/';

class ProcaSolution():
	def __init__(self, BHMass, BHSpin, ProcaMass, BosonSpin=1,CloudModel = "relativistic",units="physical", **kwargs):
			if units=="natural":
				raise NotImplementedError("natural units not yet implemented. Use physical units. See https://bitbucket.org/weast/superrad/src/master/ for definitions")
			self.SMBHMass = BHMass
			self.SMBHSpin = BHSpin
			self.ProcaMass = ProcaMass
			self.units = units
			self.BosonSpin = BosonSpin
			self.CloudModel = CloudModel

			#option to pass ultralightBoson instance to class instantiation so dont have to reinstantiate ultralightboson every time procasolution is instantiated
			if "UltralightBoson" in kwargs.keys():
				self.BosonCloud = kwargs["UltralightBoson"]
			else:
				print("UltralightBoson instance not provide. Instantiating class...")
				SR.ultralight_boson.UltralightBoson(spin=self.BosonSpin, model=self.CloudModel)
				print("done.")

			self.mode_number=1

			try:
				self.BosonWaveform = self.BosonCloud.make_waveform(BHMass, BHSpin, ProcaMass, units=units)
			except ValueError as err:
				print("Error in Proca Solution: \n {0}".format(err))

			self.Kerr = Kerr(BHSpin=BHSpin)
			self.alpha = alphavalue(self.SMBHMass, self.ProcaMass)
			self.enden = self.GetEnergyDensity()

	@property
	def FinalBHMass(self):
		return self.BosonWaveform._Mbh
	@property
	def FinalBHSpin(self):
		return self.BosonWaveform._abh

	def ChangeInOrbitalEnergy(self, SecondaryMass=1, SMBHMass=1):
		MassRatio = SecondaryMass/SMBHMass
		fractionalenden = self.FractionalGWEFlux() #anonymous function in (t,p)
		deltaEdeltaM = self.Kerr.dEdM() #anonymous function in (e,p)
		DeltaOrbitalEnergy = lambda t,e,p: MassRatio*deltaEdeltaM(e,p)*fractionalenden(t,p)

		return DeltaOrbitalEnergy

	def FractionalGWEFlux(self):
		FractionalFactor = lambda p:self.FractionalEnergyDensity(p)**2
		FractionalEnDen = lambda t,p: FractionalFactor(p)*self.BosonCloudGWEFlux(t)

		return FractionalEnDen

	def FractionalGWLFlux(self):
		fracenden = self.FractionalGWEFlux()
		factor = lambda t: self.BosonWaveform.azimuthal_num()/(self.BosonWaveform.freq_gw(t)*unit.Hz).decompose()

		ret = lambda t,p: fracenden(t,p)*factor(t)
		return ret

	def FractionalEnergyDensity(self, r):
		rstart = self.radial_data[0]
		rmax = self.radial_data[-1]
		thstart = self.theta_data[0]
		thstop = self.theta_data[-1]
		toten = lambda r: self.enden.integral(rstart, r, thstart, thstop)
		val = toten(r)/toten(rmax)
		return val

	def GetEnergyDensity(self,mode=1,overtone=0):
		enden = self._generate_interp(self.alpha,mode=1,overtone=0)

		return enden

	def _get_closest_alpha_data(self, alpha,mode=1,overtone=0):


		#extract alpha value from filename
		allfilenames = np.array(os.listdir(pathToSolutionSet))

		modeovertonedata = allfilenames[[i[-21:-4]=="Mode_"+str(1)+"_Overtone_"+str(0) for i in allfilenames]]

		alpha_regex = re.compile('Alpha_\d+_\d+')
		alpha_matches = [alpha_regex.findall(string) for string in modeovertonedata]
		value_regex = re.compile('\d+')
		numdoms = [list(map(float, value_regex.findall(st[0]))) for st in alpha_matches]
		alphas = [i[0]/i[1] for i in numdoms]

		#find closest alpha value to input parameter
		tmp = np.abs(alpha-alphas)
		self.MatchedAlpha = alpha-tmp.min()
		indicies = np.where(tmp==tmp.min())
		result = np.array(modeovertonedata)[indicies][0]

		#import data
		data = np.load(pathToSolutionSet+result)

		return data

	def _generate_interp(self, alpha,mode=1,overtone=0):
		data = self._get_closest_alpha_data(alpha,mode=mode, overtone=overtone)

		self.radial_data = data["RadialData"]
		self.theta_data = data["ThetaData"]
		energy_data = data["EnergyData"]


		interp = spint.RectBivariateSpline(self.radial_data, self.theta_data, energy_data)
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
		return (azimuthalnumber*Eflux/frequency).decompose() #all power emitted in single proca mode. In units of cloud energy Mc * c^2

	def BosonCloudGWTimescale(self):
		return self.BosonWaveform.gw_time()

	def BosonCloudMass(self,t=0):
		return self.BosonWaveform.mass_cloud(t)

	def superradiant_condition(self):
		self.proca_frequency = self.BosonCloud._cloud_model.omega_real(self.mode_number,self.alpha,self.SMBHSpin,0)/self.SMBHMass
		horfreq = self.Kerr.Horizon_Frequency()/self.SMBHMass

		return self.proca_frequency<self.mode_number*horfreq





















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
