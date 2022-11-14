from scipy.integrate import DOP853
from mpmath import *
mp.dps=25
mp.pretty=True

import numpy as np
from ..Utils import *
from ..UndressedFluxes import *

from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.utils.utility import *
from few.waveform import AAKWaveformBase
from few.summation.aakwave import AAKSummation


def Power(x,n):
	return x**n

def Sqrt(x):
	return x**(1/2)
def Sign(x):
	return x/abs(x)

class PN(Kerr, FluxFunction):
	def __init__(self, M,m, bhspin=0.9, DeltaEFlux=0.0, DeltaLFlux=0.0, FluxName="analytic"):
		Kerr.__init__(self,BHSpin=bhspin) ###better to use super? How with multiple inheritance and multilpe arguments to inits?
		FluxFunction.__init__(self, name=FluxName)

		self.epsilon=m/M
		self.SMBHMass = M
		self.SecondaryMass = m
		self.a = bhspin
		self.RadialFrequency = self.OmegaR()
		self.AzimuthalFrequency = self.OmegaPhi()
		self.PolarFrequency = self.OmegaTheta()
		self.FluxName = FluxName
		self.UndressedEFlux = lambda e,p: self.EFlux(self.a,e,p)
		self.UndressedLFlux = lambda e,p: self.LFlux(self.a,e,p)
		self.EFluxModification = DeltaEFlux
		self.LFluxModification = DeltaLFlux


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
		if ecc>=1.0  or (semimaj-get_separatrix(self.a, ecc,1.)) < 0.1 or ecc<0:
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
		except TypeError:
			print("ERROR: type error in frequency and flux generation as (e,p)=({0},{1})".format(ecc,semimaj))

		#factor of epsilon ensures correct scaling of pdot and edot in mass ratio (see: http://arxiv.org/abs/gr-qc/0702054, eq 4.3)
		Edot = epsilon*EdotN + self.EFluxModification
		Ldot = epsilon*LdotN + self.LFluxModification

		norm = self.dLdp()(ecc,semimaj)*self.dEde()(ecc,semimaj) - self.dLde()(ecc,semimaj)*self.dEdp()(ecc,semimaj)

		pdot = (self.dEde()(ecc,semimaj)*Ldot - self.dLde()(ecc,semimaj)*Edot)/norm

		if ecc<10**(-5):
			edot=0
		else:
			edot = (self.dLdp()(ecc,semimaj)*Edot - self.dEdp()(ecc,semimaj)*Ldot)/norm

		#rate of change of azimuthal phase
		Phi_phi_dot = Omega_phi

		#rate of change of radial phase
		Phi_r_dot =  Omega_r

		dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]

		return dydt

class PNTraj(TrajectoryBase):

	def __init__(self, *args, **kwargs):
		self.DeltaEFlux = kwargs.get("DeltaEFlux", 0)
		self.DeltaLFlux = kwargs.get("DeltaLFlux", 0)
		self.FluxName = kwargs.get("FluxName","analytic")
		pass

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
		while integrator.t < T and run:
			integrator.step()
			p, e, Phi_phi, Phi_r = integrator.y
			t_out.append(integrator.t * Msec)
			p_out.append(p)
			e_out.append(e)
			Phi_phi_out.append(Phi_phi)
			Phi_r_out.append(Phi_r)

			#catch separatrix crossing and halt integration
			if (p - get_separatrix(a,e,x0))<0.1:
				run=False
				exit_reason="Passed separatrix"

			if e<0 or e>=1:
				run=False
				exit_reason="Ecccentricity exceeded bounds"

		#read data
		t = np.asarray(t_out)
		p = np.asarray(p_out)
		e = np.asarray(e_out)
		Phi_phi = np.asarray(Phi_phi_out)
		Phi_r = np.asarray(Phi_r_out)

		#add polar data
		Phi_theta = Phi_phi.copy()
		x = np.ones_like(Phi_theta)

		return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)




class EMRIWaveform(AAKWaveformBase):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):

        AAKWaveformBase.__init__(
            self,
            PNTraj,  #Trajectory class
            AAKSummation, #Summation module for combining amplitude and phase information. This generates the waveform. See: https://bhptoolkit.org/FastEMRIWaveforms/html/user/sum.html#module-few.summation.aakwave
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )
