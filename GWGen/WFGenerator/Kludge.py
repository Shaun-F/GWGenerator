from scipy.integrate import solve_ivp
from mpmath import *
mp.dps=25
mp.pretty=True
import inspect

import numpy as np
from ..Utils import *
from ..UndressedFluxes import *

from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.utils.utility import *
from few.waveform import AAKWaveformBase
from few.summation.aakwave import AAKSummation
from few.utils.baseclasses import TrajectoryBase

import astropy.units as unit
import astropy.constants as cons

SEPARATRIXDELTA=0.2;
KGtoMsun = unit.kg.to(unit.Msun)
NegativeEccentricityThreshold=-1e-3

class PN(Kerr, FluxFunction):

	def __init__(self, M,m, bhspin=0.9, DeltaEFlux=0.0*unit.kg*unit.m**2/(unit.s**3), DeltaLFlux=0.0*unit.kg*unit.m**2/(unit.s**2), FluxName="analytic"):
		Kerr.__init__(self,BHSpin=bhspin) ###better to use super()? How with multiple inheritance and multilpe arguments to inits?
		FluxFunction.__init__(self, name=FluxName)

		#convert delta fluxes to anonymous functions and discard units
		if isinstance(DeltaEFlux, unit.quantity.Quantity):
			EFluxUnit = unit.kg*unit.m**2/(unit.s**3)
			val = DeltaEFlux.to(EFluxUnit).value
			DeltaEFlux = lambda t,e,p: val
		if isinstance(DeltaLFlux, unit.quantity.Quantity):
			LFluxUnit = unit.kg*unit.m**2/(unit.s**2)
			val = DeltaLFlux.to(LFluxUnit).value
			DeltaLFlux = lambda t,e,p: val

		#sanity checks
		assert inspect.isfunction(DeltaEFlux), "Error: Delta E Flux is not a function or an astropy.unit.quantity.Quantity instance. Must be a function with argument (t,e,p) or an astropy.unit.quantity.Quantity instance"
		assert inspect.isfunction(DeltaLFlux), "Error: Delta L Flux is not a function or an astropy.unit.quantity.Quantity instance. Must be a function with argument (t,e,p) or an astropy.unit.quantity.Quantity instance"

		self.SMBHMass = M
		self.SecondaryMass = m
		self.epsilon = self.SecondaryMass/self.SMBHMass
		self.a = bhspin
		self.OrbitFrequencies = self.OrbitalFrequencies()
		self.FluxName = FluxName

		"""
		Note: Fluxes must be with respect to dimensionless time.
		"""

		#Below undressed fluxes are already in geometric units and expressed as derivatives w.r.t gravitational time. Multiply by mass ratio to get full expression as
		#	 analytic expressions in src code have mass dependence factored out
		#see, e.g., phys rev D 66, 044002  page 16   or PTEP 2015, 073E03 page 28-29
		self.UndressedEFlux = lambda e,p: self.EFlux(self.a,e,p)*self.epsilon**2 * self.SMBHMass # eq. 128 of phys rev D 66, 044002 and convert to gravitational time
		self.UndressedLFlux = lambda e,p: self.LFlux(self.a,e,p)*self.epsilon*self.SecondaryMass*self.SMBHMass #eq. 129 of "
		self.UndressedpFlux = lambda e,p: self.pFlux(self.a,e,p)*self.epsilon #eq. 137 of "
		self.UndressedeFlux = lambda e,p: self.eFlux(self.a,e,p)*self.epsilon #eq. 138 of "


		#dimensionless
		#Unit of DeltaEFlux and DeltaLFlux must be SI units (but still floats, not astropy quantity objects)
		self.InverseEnergyFlux = (cons.G/(cons.c**5)).to(unit.s**3/(unit.kg*unit.m**2)).value #G/c**5
		self.InverseAngularMomentumFlux = (1/(cons.c**2)).to((unit.s**2)/(unit.m**2)).value # (distance in units of SMBH Radius)

		self.EFluxModification = lambda t,e,p: DeltaEFlux(t,e,p)*self.InverseEnergyFlux*self.SMBHMass #convert time to units of SMBH gravitational time and SI units to geometric units
		self.LFluxModification = lambda t,e,p: DeltaLFlux(t,e,p)*KGtoMsun*self.InverseAngularMomentumFlux*self.SMBHMass #convert time to units of SMBH gravitational time and SI units to geometric units


		self.IntegratorRun=True
		self.IntegratorExitReason=""

		#unit conversions
		##semi-latus rectum is dimensionless w.r.t smbh mass
		self.dLdpUnit = (self.SecondaryMass*self.SMBHMass)#*unit.M_sun*cons.c/self.SMBHMass).decompose()
		self.dLdeUnit = (self.SecondaryMass*self.SMBHMass)#*unit.Msun*cons.G*unit.Msun/cons.c).decompose()
		self.dEdpUnit = (self.SecondaryMass) #*cons.c**4/cons.G/self.SMBHMass).decompose()
		self.dEdeUnit = (self.SecondaryMass)#*unit.Msun*cons.c**2).decompose()

		self.__SEPARATRIX=6+SEPARATRIXDELTA
		self.__SEPARATRIX_CUT =	self.__SEPARATRIX+SEPARATRIXDELTA

	@property
	def separatrix_cutoff(self):
		return self.__SEPARATRIX_CUT

	@separatrix_cutoff.setter
	def separatrix_cutoff(self, newval):
		self.__SEPARATRIX_CUT=newval

	@property
	def pdotN(self):
		return self.__pdotN

	@property
	def edotN(self):
		return self.__edotN

	def __call__(self, t, y):
		"""
		y is array holding parameters to integrate y=[p,e,Phi_phi, Phi_r]
		available kwargs:
			a: dimensionless black hole spin
		"""

		#mass ratio
		epsilon = self.epsilon

		#extract parameters to evolve
		semimaj = float(y[0])
		ecc = float(y[1])
		phi_phase = float(y[2])
		radial_phase = float(y[3])
		if ecc<1e-6:
			#if eccentricity is zero, replace it by small number to guard against poles in integrals of motion
			ecc=1e-6

		#setup guard for bad integration steps
		if ecc>=1.0  or ecc<0 or semimaj<get_separatrix(self.a,ecc,1.):
			return np.zeros_like(y)


		try:
			# Orbital Frequency
			orb_freqs = self.OrbitFrequencies(ecc,semimaj,1);
			Omega_phi = orb_freqs["OmegaPhi"];

			# Radial Frequency
			Omega_r = orb_freqs["OmegaR"];

			# Polar Frequency
			Omega_theta = orb_freqs["OmegaTheta"]

			#semilatus rectum flux
			## set fluxes to instance attributes for integration termination events
			self.__pdotN = self.UndressedpFlux(ecc,semimaj) #this is negative

			#Eccentricity flux
			self.__edotN = self.UndressedeFlux(ecc,semimaj) #this is negative

			#Energy correction
			Ecorr = self.EFluxModification(t*self.SMBHMass*MTSUN_SI, ecc, semimaj)

			#Angular Momentum Corrector
			Lcorr = self.LFluxModification(t*self.SMBHMass*MTSUN_SI, ecc, semimaj)
		except TypeError:
			print("ERROR: type error in frequency and flux generation as (e,p)=({0},{1})".format(ecc,semimaj))
		except SystemError as errmsg:
			print("Error at parameter point (p,e)=({0},{1}). \n {2}".format(semimaj,ecc,errmsg))
			self.IntegratorRun=False
			self.IntegratorExitReason=errmsg
			return np.zeros_like(y)

		if self.__pdotN>0:
			self.IntegratorRun=False
			self.IntegratorExitReason="PN Semilatus Rectum flux larger than zero! Breaking."
		elif self.__edotN>0:
			self.IntegratorRun=False
			self.IntegratorExitReason="PN Eccentricity flux larger than zero! Breaking."


		pdotN = self.__pdotN
		edotN = self.__edotN

		#(see: http://arxiv.org/abs/gr-qc/0702054, eq 4.3)
		dldp = self.dLdp()(ecc,semimaj)*self.dLdpUnit
		dlde = self.dLde()(ecc,semimaj)*self.dLdeUnit
		dedp = self.dEdp()(ecc,semimaj)*self.dEdpUnit
		dede = self.dEde()(ecc,semimaj)*self.dEdeUnit
		norm = (dldp*dede - dlde*dedp)



		if ecc<=1e-6:
			edot=0
			pdotcorr = Ecorr/dedp
			pdot = pdotN + pdotcorr
		else:
			pdotCorr = (1/norm)*(dede*Lcorr - dlde*Ecorr)
			edotCorr = (1/norm)*(dldp*Ecorr - dedp*Lcorr)
			edot = edotN + edotCorr
			pdot = pdotN + pdotCorr

		#rate of change of azimuthal phase
		Phi_phi_dot = Omega_phi

		#rate of change of radial phase
		Phi_r_dot =  Omega_r

		#rate of change of polar phase
		Phi_theta_dot = Omega_theta

		dydt = [pdot, edot, Phi_phi_dot, Phi_theta_dot, Phi_r_dot]

		return dydt

class PNTraj(TrajectoryBase):
	def __init__(self,**kwargs):
		self.__exit_reason = ""


	def get_inspiral(self, M, mu, a, p0, e0, x0, T=1.0, **kwargs):
		"""
		M: mass of central SMBH
		mu: mass of orbiting CO
		a: dimensionless spin of SMBH
		p0: initial semi-latus rectum
		e0: initial eccentricity (NOTE: currently only considering circular orbits
		x0: initial inclination of orbital plane (NOTE: currently only considering equatorial orbits)
		T: integration time (years)
		"""





		self.__integration_method = kwargs.get("integration_method","DOP853")
		self.__dense_output = kwargs.get("dense_output", True)
		self.__SEPARATRIX_DELTA = kwargs.get("SEPARATRIX_DELTA", SEPARATRIXDELTA)
		npoints = kwargs.get("npoints", 100)

		self.DeltaEFlux = kwargs.get("DeltaEFlux", 0.0*unit.kg*unit.m**2/(unit.s**3))
		self.DeltaLFlux = kwargs.get("DeltaLFlux", 0.0*unit.kg*unit.m**2/(unit.s**2))
		self.FluxName = kwargs.get("FluxName","analytic")
		self.SMBHMass = M

		assert float(x0)==1., "Error: Only equatorial orbits are currently implemented."
		#boundary values
		if e0<10**(-10): #guard against poles in analytic expressions
			e0=10**(-10)
		y0 = [p0, e0, 0.0, 0.0, 0.0] #zero mean anomaly initially

		#compute separatrix of initial parameters
		self.__initial_separatrix = get_separatrix(float(a), 0., 1.)
		self.__SEPARATRIX_CUTOFF = self.__initial_separatrix + self.__SEPARATRIX_DELTA


		#MTSUN_SI converts solar masses to seconds and is equal to G/(c^3)
		#YRSID_SI converts years into seconds
		t_start = 0
		t_stop = T * YRSID_SI / (M * MTSUN_SI) #dimensionless time
		t_res = t_stop/npoints

		SMBHSeconds = M*MTSUN_SI

		#PN evaluator
		epsilon = float(mu/M)
		self.PNEvaluator = PN(M,mu,bhspin=a, DeltaEFlux = self.DeltaEFlux, DeltaLFlux = self.DeltaLFlux, FluxName=self.FluxName)
		self.PNEvaluator.separatrix_cutoff = self.__SEPARATRIX_CUTOFF

		# run integrator down to T or separatrix
		t_span = (t_start, t_stop)

		def __integration_event_tracker_eccentricity(_, y_vec):
			e = y_vec[1]
			#define a function which is has a zero at e=1, a zero at the smallest negative float, and positive on the range [0,1)
			eps = np.finfo(float).eps
			x_shift = (1-eps**2)/(2*(eps+1))
			y_shift = (eps+x_shift)**2
			res = y_shift - (e-x_shift)**2
			if res<=0:
				self.__exit_reason = "Eccentricity exceeded bounds"
			return res

		def __integration_event_tracker_semilatus_rectum(_, y_vec):
			p = y_vec[0]
			res = p-get_separatrix(a,y_vec[1],1.)
			if res<=0:
				self.__exit_reason = "Separatrix reached!"
			return res

		def __integration_event_tracker_eFlux(_, y_vec):
			Eflux = self.PNEvaluator.UndressedeFlux(y_vec[1], y_vec[0])
			res = -Eflux
			if res<=0:
				self.__exit_reason="PN Eccentricity flux larger than zero! Breaking."
			return res
		def __integration_event_tracker_pFlux(_, y_vec):
			Lflux = self.PNEvaluator.UndressedpFlux(y_vec[1], y_vec[0])
			res = -Lflux
			if res<=0:
				self.__exit_reason="PN Semilatus Rectum flux larger than zero! Breaking."
			return res

		__integration_event_tracker_eccentricity.terminal=True
		__integration_event_tracker_semilatus_rectum.terminal=True
		__integration_event_tracker_eFlux.terminal=True
		__integration_event_tracker_pFlux.terminal=True

		self.__integration_event_trackers = [#__integration_event_tracker_eccentricity,
										__integration_event_tracker_semilatus_rectum,
										__integration_event_tracker_pFlux]

		max_step_size = t_span[-1]/npoints

		result = solve_ivp(self.PNEvaluator,
							t_span, #time range
							y0, #initial values
							method=self.__integration_method, #integration method
							dense_output=self.__dense_output, #compute interpolation over points
							events = self.__integration_event_trackers, #track boundaries of integration
							max_step = max_step_size #dimensionless seconds
						)

		#check eccentricity bounds
		assert np.all(result['y'][1]>=NegativeEccentricityThreshold), "Error: Eccentricity outside tolerable negative value range."
		t_out = result["t"]
		p_out = result["y"][0]
		e_out = result['y'][1]*(result["y"][1]>=0) #Cast negative values to 0. ALready confirmed negatives values are greater than threshold
		e_out[e_out==0]=1e-10 #cast the zero values to small positive values to avoid poles in internal functions
		Phi_phi_out = result["y"][2]
		Phi_theta_out = result["y"][3]
		Phi_r_out = result["y"][4]

		if self.__exit_reason=="":
			self.__exit_reason = "Integration reached time boundary. Boundary location t = {0:0.2f}".format(t_out[-1])

		if self.__dense_output:
			if npoints<=len(t_out):
				npoints = len(t_out)+50
			new_time_domain = np.array(IncreaseArrayDensity(t_out,npoints))

			interpolationfunction = result["sol"]
			new_data = interpolationfunction(new_time_domain)
			t_out = new_time_domain
			p_out, e_out, Phi_phi_out, Phi_theta_out, Phi_r_out = new_data
			e_out = e_out*(e_out>=0)
			e_out[e_out==0]=1e-10 #cast zero values to small positive values to avoid

		#add polar data
		#### Only equatorial orbits implemented.
		x = np.ones_like(Phi_theta_out)

		#cast to array objects compatible with c code
		t = ConvertToCCompatibleArray(t_out*SMBHSeconds,newdtype=np.float64)
		p = ConvertToCCompatibleArray(p_out,newdtype=np.float64)
		e = ConvertToCCompatibleArray(e_out,newdtype=np.float64)
		Phi_phi = ConvertToCCompatibleArray(Phi_phi_out,newdtype=np.float64)
		Phi_theta = ConvertToCCompatibleArray(Phi_theta_out,newdtype=np.float64)
		Phi_r = ConvertToCCompatibleArray(Phi_r_out,newdtype=np.float64)

		return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)

	#mutable properties
	@property
	def integration_method(self):
		return self.__integration_method
	@integration_method.setter
	def integration_method(self,newmeth):
		self.__integration_method=newmeth

	@property
	def dense_output(self):
		return self.__dense_output
	@dense_output.setter
	def dense_output(self,newmeth):
		self.__dense_output=newmeth


	@property
	def separatrix_delta(self):
		return self.__SEPARATRIX_DELTA
	@separatrix_delta.setter
	def separatrix_delta(self,newval):
		self.__SEPARATRIX_DELTA=newval



	#immutable properties
	@property
	def exit_reason(self):
		return self.__exit_reason

	@property
	def separatrix_cut(self):
		if hasattr(self, "_PNTraj__SEPARATRIX_CUTOFF"):
			return self.__SEPARATRIX_CUTOFF
		else:
			print("Run trajectory method to generate this property")
			return None


class EMRIWaveform(AAKWaveformBase):
	def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):
		self.inspiralkwargs = inspiral_kwargs.copy()
		self.sumkwargs = sum_kwargs.copy()
		self.use_gpu = use_gpu
		self.num_threads = num_threads

		self.sanity_check_gpu(self.use_gpu)
		#added a class method __call__ with should be run with
		## EMRIWaveform()(SMBHMass, SecondaryMass, BHSpin, p0, e0, x0, qs,phis,qk,phik, dist, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
		AAKWaveformBase.__init__(self,
								PNTraj,  #Trajectory class
            					AAKSummation, #Summation module for combining amplitude and phase information. This generates the waveform. See: https://bhptoolkit.org/FastEMRIWaveforms/html/user/sum.html#module-few.summation.aakwave
            					inspiral_kwargs=self.inspiralkwargs,
            					sum_kwargs=self.sumkwargs,
            					use_gpu=self.use_gpu,
            					num_threads=self.num_threads
	            				)


	def __call__(self, SMBHMass, SecondaryMass, BHSpin, p0, e0, x0, qs, phis, qk, phik,dist, T=1, npoints=100, FluxName="analytic", **kwargs):
		Phi_phi0 = kwargs.get("Phi_phi0", 0)
		Phi_theta0 = kwargs.get("Phi_theta0",0)
		Phi_r0 = kwargs.get("Phi_r0", 0)
		mich = kwargs.get("mich", False)
		dt = kwargs.get("dt", 15)
		if e0<1e-6:
			warnings.warn("Eccentricity below safe threshold for FEW. Functions behave poorly for e<1e-6. Enforcing e=1e-6 for all further computations")
			e0=1e-6 #Certain functions in FEW are not well-behaved below this value

		qS,phiS,qK,phiK = self.sanity_check_angles(qs,phis,qk,phik)
		self.sanity_check_init(SMBHMass, SecondaryMass,BHSpin,p0,e0,x0)

		#get the Trajectory
		t,p,e,Y,pphi,ptheta,pr = self.inspiral_generator(SMBHMass,SecondaryMass,BHSpin,p0,e0,x0,T=T, dt=dt, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, **self.inspiralkwargs)
		self.Trajectory = {"t":t, "p":p, "e":e, "Y":Y, "Phi_phi":pphi, "Phi_theta":ptheta, "Phi_r":pr}
		self.sanity_check_traj(p,e,Y)
		self.end_time = t[-1]

        # number of modes to use (from original AAK model)
		self.num_modes_kept = self.nmodes = int(30 * e0)
		if self.num_modes_kept < 4:
			self.num_modes_kept = self.nmodes = 4

		self.waveform = self.create_waveform(t,SMBHMass,BHSpin,p,e,Y,pphi, ptheta, pr, SecondaryMass,qS,phiS, qK, phiK, dist, self.nmodes,mich=mich,dt=dt,T=T)
		return self.waveform
