import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate as spint
import os
import superrad as SR
from superrad import ultralight_boson
from GWGen.Utils import *
import re
from bisect import bisect_right

pathToSolutionSet = os.path.abspath(os.path.dirname(__file__))+'/../ProcaData/';
RadialDataTruncationFactor = 7

class ProcaSolution():
	def __init__(self, BHMass, BHSpin, ProcaMass, BosonSpin=1,CloudModel = "relativistic",units="physical", UltralightBoson=None,**kwargs):
			if units=="natural":
				raise NotImplementedError("natural units not yet implemented. Use physical units. See https://bitbucket.org/weast/superrad/src/master/ for definitions")
			self.SMBHMass = BHMass
			self.SMBHSpin = BHSpin
			self.ProcaMass = ProcaMass
			self.units = units
			self.BosonSpin = BosonSpin
			self.CloudModel = CloudModel
			self.BosonClass = UltralightBoson

			#option to pass ultralightBoson instance to class instantiation so dont have to reinstantiate ultralightboson every time procasolution is instantiated
			if self.BosonClass==None:
				print("UltralightBoson instance not provided. Instantiating class...")
				self.BosonClass=SR.ultralight_boson.UltralightBoson(spin=self.BosonSpin, model=self.CloudModel)
				print("done.")

			self.mode_number=1

			try:
				self.BosonWaveform = self.BosonClass.make_waveform(BHMass, BHSpin, ProcaMass, units=units)
			except ValueError as err:
				raise RuntimeError("Error in Proca Solution: \n {0}".format(err))

			self.Kerr = Kerr(BHSpin=BHSpin)
			self.alpha = alphavalue(self.SMBHMass, self.ProcaMass)
			self.enden = self.GetEnergyDensity()

	@property
	def FinalBHMass(self):
		return self.BosonWaveform._Mbh
	@property
	def FinalBHSpin(self):
		return self.BosonWaveform._abh

	def ChangeInOrbitalConstants(self, SecondaryMass=1, SMBHMass=1):
		MassRatio = SecondaryMass/SMBHMass
		fractionalEnergyFlux = self.FractionalGWEFlux() #anonymous function in (t,p)
		fractionalAngularMomentumFlux = self.FractionalGWLFlux() #anonymous function in (t,p)
		deltaEdeltaM = self.Kerr.dEdM() #anonymous function in (e,p)
		deltaEdeltaa = self.Kerr.dEda() #anonymous function in (e,p)
		deltaLdeltaM = self.Kerr.dLdM() #anonymous function in (e,p)
		deltaLdeltaa = self.Kerr.dLda() #anonymous function in (e,p)
		#Secondary Mass prefactor comes from expression for energy and angular momentum (converting specific constants to full constants)
		#SMBH Mass inverse prefactor comes from full expression for derivatives of constants of motion
		DeltaOrbitalEnergy = lambda t,e,p: MassRatio*(deltaEdeltaM(e,p)*fractionalEnergyFlux(t,p) + deltaEdeltaa(e,p)*fractionalAngularMomentumFlux(t,p))
		DeltaOrbitalAngularMomentum = lambda t,e,p: MassRatio*(deltaLdeltaM(e,p)*fractionalEnergyFlux(t,p) + deltaLdeltaa(e,p)*fractionalAngularMomentumFlux(t,p))

		res = {"E": lambda t,e,p: DeltaOrbitalEnergy(t,e,p),"L": lambda t,e,p: DeltaOrbitalAngularMomentum(t,e,p)}
		return res


	def FractionalGWEFlux(self):
		FractionalFactor = lambda p:self.FractionalEnergyDensity(p)**2
		FractionalEnDen = lambda t,p: FractionalFactor(p)*self.BosonCloudGWEFlux(t)

		return FractionalEnDen

	def FractionalGWLFlux(self):
		fracenden = self.FractionalGWEFlux()
		factor = lambda t: self.BosonWaveform.azimuthal_num()/(self.BosonWaveform.freq_gw(t))

		ret = lambda t,p: fracenden(t,p)*factor(t)
		return ret

	def FractionalEnergyDensity(self, r):
		"""
			Fraction of total energy within a given radius
		"""
		rstart = self.radial_data[0]
		rmax = self.radial_data[-1]
		thstart = self.theta_data[0]
		thstop = self.theta_data[-1]
		toten = lambda r: self.enden.integral(rstart, r, thstart, thstop)
		val = toten(r)/toten(rmax)
		return val

	def GetEnergyDensity(self,mode=1,overtone=0):
		try:
			enden = self._generate_interp(self.alpha,mode=1,overtone=0)
		except ValueError as err:
			errmessage = "Error generating energy density: \n\t SMBHMass {0} \n\t Proca Mass {1}  \n\t Error Message {2}".format(self.SMBHMass, self.ProcaMass, err.args[0])
			raise ValueError(errmessage)

		return enden

	def _get_closest_alpha_datasets(self, alpha,mode=1,overtone=0):


		#import filenames
		allfilenames = glob.glob(pathToSolutionSet+"BHSpin*")
		modeovertonebool = [bool(re.search("Mode_1_Overtone_0", i)) for i in allfilenames]
		newallfilenames=[]
		for inx, boolval in enumerate(modeovertonebool):
		    if boolval:
		        newallfilenames.append(allfilenames[inx])
		allfilenames = newallfilenames

		#sort filenames
		unsorted_alphavalues = list(map(AlphaValFromFilename, allfilenames))
		index_sort = np.argsort(unsorted_alphavalues)
		alphavalues = np.array(unsorted_alphavalues)[index_sort]
		allfilenames = np.array(allfilenames)[index_sort]

		#selected closest neighbors, assuming alpha values are monotonically increasing
		##First assert requested alpha value in range of available proca data
		assert np.logical_and(alpha>alphavalues[0], alpha<alphavalues[-1]), "Error: Alpha value out of range of available data."
		larger_alpha_index = bisect_right(alphavalues, alpha)
		if larger_alpha_index == 0:
			smaller_alpha_index = 0
		else:
			smaller_alpha_index = larger_alpha_index - 1
		alphaNeighborsIndex = (smaller_alpha_index, larger_alpha_index)
		selectedfilenames = [allfilenames[i] for i in alphaNeighborsIndex]
		selecteddata = [np.load(i) for i in selectedfilenames]

		#import data
		selecteddata = [np.load(i) for i in selectedfilenames]
		selectedalphas = alphavalues[[alphaNeighborsIndex[0],alphaNeighborsIndex[1]]]

		return selectedalphas, selecteddata

	def _generate_interp(self, alpha,mode=1,overtone=0):
		alphas, datas = self._get_closest_alpha_datasets(alpha,mode=mode, overtone=overtone)

		RadialDataSet = [i["RadialData"] for i in datas]
		ThetaDataSet = [i["ThetaData"][0:100] for i in datas]
		NewShape = (int(min([len(i) for i in RadialDataSet])/RadialDataTruncationFactor), 100) #(radial data array shape, theta data array shape)
		RadialDataSet = [i[:NewShape[0]] for i in RadialDataSet]
		ThetaDataSet = [i[:NewShape[1]] for i in ThetaDataSet]

		self.__RawEnergyData = [i["EnergyData"][:NewShape[0],:NewShape[1] ] for i in datas]
		#generate 3d interpolation function over the radial data, theta data, and alpha data
		interp = spint.RegularGridInterpolator((alphas, RadialDataSet[0], ThetaDataSet[0]), self.__RawEnergyData)
		coords = cartesian_product(np.array([self.alpha]), RadialDataSet[0], ThetaDataSet[0])
		try:
			InterpolatedEnergyValues = np.reshape(interp(coords), NewShape)
		except ValueError:
			print("DEBUG: Input alpha {0} \n self.alpha {1} \n alpha neighbors {2} \n datasets {3} ".format(alpha, self.alpha, alphas, datas))
			raise ValueError("Error in generating interpolation function for SMBH Mass {0}, Proca mass {1}, and alpha {2}".format(self.SMBHMass, self.ProcaMass, self.alpha))

		self.radial_data = RadialDataSet[0]
		self.theta_data = ThetaDataSet[0]

		#linear interpolation over radial data and theta data for given alpha value
		InterpolationFunction = spint.RectBivariateSpline(self.radial_data, self.theta_data, InterpolatedEnergyValues)

		return InterpolationFunction


	def BosonCloudGWEFlux(self,t=0):
		#Must divide by mass ratio m/M to get actual dimensionless power, where m is the mass of the secondary
		## See definition of dimenionless energy and dimensionless time
		DimensionlessPower = self.BosonWaveform.power_gw(t) # SI units
		return -DimensionlessPower
	def BosonCloudGWLFlux(self,t=0):
		Eflux = self.BosonCloudGWEFlux(t) #dimensionless power
		azimuthalnumber = self.BosonWaveform.azimuthal_num()
		frequency = self.BosonWaveform.freq_gw(t)
		return azimuthalnumber*Eflux/frequency #all power emitted in single proca mode. In units of cloud energy Mc * c^2

	def BosonCloudGWTimescale(self):
		return self.BosonWaveform.gw_time()

	def BosonCloudMass(self,t=0):
		return self.BosonWaveform.mass_cloud(t)

	def superradiant_condition(self):
		self.proca_frequency = self.BosonClass._cloud_model.omega_real(self.mode_number,self.alpha,self.SMBHSpin,0)/self.SMBHMass
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
