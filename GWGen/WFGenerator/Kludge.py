from scipy.integrate import DOP853
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

import astropy.units as unit;
import astropy.constants as cons

SEPARATRIXCUTOFF=0.1;

class PN(Kerr, FluxFunction):

	def __init__(self, M,m, bhspin=0.9, DeltaEFlux=0.0*unit.kg*unit.m**2/(unit.s**3), DeltaLFlux=0.0*unit.kg*unit.m**2/(unit.s**2), FluxName="analytic"):
		Kerr.__init__(self,BHSpin=bhspin) ###better to use super? How with multiple inheritance and multilpe arguments to inits?
		FluxFunction.__init__(self, name=FluxName)

		#sanity checks
		assert inspect.isfunction(DeltaEFlux), "Error: Delta E Flux is not a function. Must be a function with argument (t,e,p)"
		assert inspect.isfunction(DeltaLFlux), "Error: Delta L Flux is not a function. Must be a function with argument (t,e,p)"

		ranvals = [0.1 for i in inspect.signature(DeltaEFlux).parameters]
		ranvals[-1]=10
		assert DeltaEFlux(*ranvals).unit == unit.kg*unit.m**2/(unit.s**3), "Error: DeltaEFlux must have units kg m**2/s**3"
		assert DeltaLFlux(*ranvals).unit == unit.kg*unit.m**2/(unit.s**2), "Error: DeltaLFlux must have units kg m**2/s**2"


		self.epsilon=m/M
		self.SMBHMass = M
		self.SecondaryMass = m
		self.a = bhspin
		self.RadialFrequency = self.OmegaR()
		self.AzimuthalFrequency = self.OmegaPhi()
		self.PolarFrequency = self.OmegaTheta()
		self.FluxName = FluxName

		efluxUnitConv = (self.epsilon**2*(cons.c**5)/cons.G).to(unit.kg*unit.m**2/(unit.s**3))
		lfluxUnitConv = (self.epsilon * self.SecondaryMass * unit.Msun * cons.c**2).to(unit.kg*unit.m**2/(unit.s**2))
		self.UndressedEFlux = lambda e,p: self.EFlux(self.a,e,p)*efluxUnitConv #dimensionfull
		self.UndressedLFlux = lambda e,p: self.LFlux(self.a,e,p)*lfluxUnitConv #dimensionfull
		self.EFluxModification = DeltaEFlux
		self.LFluxModification = DeltaLFlux
		self.IntegratorRun=True
		self.IntegratorExitReason=""

		#unit conversions
		self.dldpUnit = (self.SecondaryMass*unit.Msun*cons.c).decompose()
		self.dldeUnit = (self.SecondaryMass*unit.Msun*cons.G*self.SMBHMass*unit.Msun/cons.c).decompose()
		self.dedpUnit = (self.epsilon*cons.c**4/cons.G).decompose()
		self.dedeUnit = (self.SecondaryMass*unit.Msun*cons.c**2).decompose()

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
		#setup guard for bad integration steps
		if ecc>=1.0  or (semimaj-get_separatrix(self.a, ecc,1.)) < SEPARATRIXCUTOFF or ecc<0:
			return [0.0, 0.0,0.0,0.0]

		if ecc==0.0:
			#if eccentricity is zero, replace it by small number to guard against poles in integrals of motion
			ecc=1e-16
		try:
			# Azimuthal Frequency
			Omega_phi = self.AzimuthalFrequency(ecc,semimaj);

			# Radial Frequency
			Omega_r = self.RadialFrequency(ecc,semimaj)

			#Energy flux
			EdotN = self.UndressedEFlux(ecc,semimaj) #this is negative

			#Angular momentum
			LdotN = self.UndressedLFlux(ecc,semimaj) #this is negative

			#Energy correction
			Ecorr = self.EFluxModification(t*self.SMBHMass*MTSUN_SI, ecc, semimaj)

			#Angular Momentum Corrector
			Lcorr = self.LFluxModification(t*self.SMBHMass*MTSUN_SI, ecc, semimaj)
		except TypeError:
			print("ERROR: type error in frequency and flux generation as (e,p)=({0},{1})".format(ecc,semimaj))

		#(see: http://arxiv.org/abs/gr-qc/0702054, eq 4.3)
		Edot = EdotN + Ecorr #units: kg m**2/s**3
		Ldot = LdotN + Lcorr #units: kg m**2/s**2
		dldp = self.dLdp()(ecc,semimaj)*self.dldpUnit
		dlde = self.dLde()(ecc,semimaj)*self.dldeUnit
		dedp = self.dEdp()(ecc,semimaj)*self.dedpUnit
		dede = self.dEde()(ecc,semimaj)*self.dedeUnit

		norm = (dldp*dede - dlde*dedp)
		pdot = (dede*Ldot - dlde*Edot)/norm #units: m/s

		if ecc<10**(-5):
			edot=0
		else:
			edot = (dldp*Edot - dedp*Ldot)/norm
		if EdotN>0:
			self.IntegratorRun=False
			self.IntegratorExitReason="PN Energy flux larger than zero! Breaking."
		elif LdotN>0:
			self.IntegratorRun=False
			self.IntegratorExitReason="PN Angular Momentum flux larger than zero! Breaking."

		#adimensionlize
		pdot = (pdot/cons.c).decompose().value
		edot = (edot*cons.G*self.SMBHMass*unit.Msun/(cons.c**3)).decompose().value

		#rate of change of azimuthal phase
		Phi_phi_dot = Omega_phi

		#rate of change of radial phase
		Phi_r_dot =  Omega_r

		dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]

		return dydt

class PNTraj(TrajectoryBase):

	def get_inspiral(self, M, mu, a, p0, e0, x0, T=1.0,npoints=10, **kwargs):
		"""
		M: mass of central SMBH
		mu: mass of orbiting CO
		a: dimensionless spin of SMBH
		p0: initial semi-latus rectum
		e0: initial eccentricity (NOTE: currently only considering circular orbits
		x0: initial inclination of orbital plane (NOTE: currently only considering equatorial orbits)
		T: integration time (years)
		"""
		self.DeltaEFlux = kwargs.get("DeltaEFlux", 0.0*unit.kg*unit.m**2/(unit.s**3))
		self.DeltaLFlux = kwargs.get("DeltaLFlux", 0.0*unit.kg*unit.m**2/(unit.s**2))
		self.FluxName = kwargs.get("FluxName","analytic")
		self.SMBHMass = M
		#boundary values
		y0 = [p0, e0, 0.0, 0.0] #zero mean anomaly initially


		#MTSUN_SI converts solar masses to seconds and is equal to G/(c^3)
		#YRSID_SI converts years into seconds

		T = T * YRSID_SI / (M * MTSUN_SI)

		Msec = M*MTSUN_SI

		#PN evaluator
		epsilon = float(mu/M)
		self.PNEvaluator = PN(M,mu,bhspin=a, DeltaEFlux = self.DeltaEFlux, DeltaLFlux = self.DeltaLFlux, FluxName=self.FluxName)
		integrator = DOP853(self.PNEvaluator, 0.0, y0, T, max_step=T/npoints) #Explicit Runge-Kutta of order 8
		#arrays to hold output values from integrator
		t_out, p_out, e_out = [0.], [p0], [e0]
		Phi_phi_out, Phi_r_out = [0.], [0.]

		# run integrator down to T or separatrix
		run=True
		exit_reason=""
		while integrator.t < T and run:
			integrator.step()
			p, e, Phi_phi, Phi_r = integrator.y
			t_out.append(integrator.t * Msec)
			p_out.append(p)
			e_out.append(e)
			Phi_phi_out.append(Phi_phi)
			Phi_r_out.append(Phi_r)

			#catch separatrix crossing and halt integration
			if (p - get_separatrix(float(a),float(e),float(x0)))<SEPARATRIXCUTOFF:
				run=False
				exit_reason="Passed separatrix"

			if e<0 or e>=1:
				run=False
				exit_reason="Ecccentricity exceeded bounds"
			if p_out[-1]>p_out[-2]:
				print("Error: semi-latus rectum increased! Breaking")
				run=False
				exit_reason="Semi-latus rectum increased. Possibly beyond validity of PN expressions for fluxes"

			#catch breaks from integrand function
			run=self.PNEvaluator.IntegratorRun
			exit_reason=self.PNEvaluator.IntegratorExitReason


		if exit_reason!="":
			print("Integration halted before ending time. Reasons: {0}".format(exit_reason))

		#read data
		t = np.asarray(t_out)
		p = np.asarray(p_out)
		e = np.asarray(e_out)
		Phi_phi = np.asarray(Phi_phi_out)
		Phi_r = np.asarray(Phi_r_out)

		#add polar data
		Phi_theta = (0)*np.ones_like(Phi_phi)
		x = np.ones_like(Phi_theta)

		return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)




class EMRIWaveform(AAKWaveformBase):
	def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):
		self.inspiralkwargs = inspiral_kwargs
		self.sumkwargs = sum_kwargs
		self.use_gpu = use_gpu
		self.num_threads = num_threads
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


	def __call__(self, SMBHMass, SecondaryMass, BHSpin, p0, e0, x0, qs, phis, qk, phik,dist, T=1, npoints=10, BosonSpin=1, CloudModel="relativistic", units="physical", FluxName="analytic", **kwargs):
		massRatio = SecondaryMass/SMBHMass
		Phi_phi0 = kwargs.get("Phi_phi0", 0)
		Phi_theta0 = kwargs.get("Phi_theta0",0)
		Phi_r0 = kwargs.get("Phi_r0", 0)
		mich = kwargs.get("mich", False)
		dt = kwargs.get("dt", 15)
		if e0<1e-6:
			warnings.warn("Eccentricity below safe threshold for FEW. Functions behave poorly for e<1e-6")
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
