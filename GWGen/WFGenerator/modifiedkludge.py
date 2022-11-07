from scipy.integrate import DOP853
import numpy as np
from .Kerr import *
from GWGen.UndressedFluxes import *

from few.utils.baseclasses import TrajectoryBase
from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
from few.waveform import AAKWaveformBase
from few.summation.aakwave import AAKSummation


def Power(x,n):
	return x**n

def Sqrt(x):
	return x**(1/2)
def Sign(x):
	return x/abs(x)

class PN(Kerr):
	def __init__(self, massratio, bhspin=0.9): #epsilon is mass ratio
		self.epsilon=massratio
		self.a = bhspin
		super().__init__(BHSpin=self.a)
		self.RadialFrequency = self.OmegaR()
		self.AzimuthalFrequency = self.OmegaPhi()
		self.PolarFrequency = self.OmegaTheta()
		self.UndressedFlux = lambda e,p: Analytic5PNFlux(e,p,self.a)


	def __call__(self, t, y):
		"""
		y is array holding parameters to integrate y=[p,e,Phi_phi, Phi_r]
		available kwargs:
			a: dimensionless black hole spin
		"""


		#mass ratio
		epsilon = self.epsilon

		#extract parameters to evolve
		p, e, Phi_phi, Phi_r = y


		#setup guard for bad integration steps
		if e>=1.0 or e<1e-3 or p<6.0 or (p - 6 - 2* e) < 0.1:
			return [0.0, 0.0,0.0,0.0]

		# Azimuthal Frequency
		Omega_phi = self.AzimuthalFrequency(e,p);

		# Radial Frequency
		Omega_r = self.RadialFrequency(e,p)

		#Energy flux
		EdotN = self.UndressedFlux(e,p);

		#include mass ratio
		Edot = epsilon*EdotN
		Ldot = Edot/Omega_phi ########### Check this expression. Note it holds for circular orbits

		pdot = Edot/(self.dEdp()(e,p)) + Ldot/(self.dLdp()(e,p))

		edot = Edot/(self.dEde()(e,p)) + Ldot/(self.dLde()(e,p))

		#rate of change of azimuthal phase
		Phi_phi_dot = Omega_phi

		#rate of change of radial phase
		Phi_r_dot =  Omega_r

		dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]

		return dydt

class PnTraj(TrajectoryBase):

	def __init__(self, *args, **kwargs):
		pass

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
		#boundary values
		y0 = [p0, e0, 0.0, 0.0] #zero mean anomaly initially


		#MTSUN_SI converts solar masses to seconds and is equal to G/(c^3)
		#YRSID_SI converts years into seconds
		T = T * YRSID_SI / (M * MTSUN_SI)

		Msec = M*MTSUN_SI

		epsilon = mu/M
		integrator = DOP853(PN(epsilon), 0.0, y0, T) #Explicit Runge-Kutta of order 8

		#arrays to hold output values from integrator
		t_out, p_out, e_out = [], [], []
		Phi_phi_out, Phi_r_out = [], []
		t_out.append(0.0)
		p_out.append(p0)
		e_out.append(e0)
		Phi_phi_out.append(0.0)
		Phi_r_out.append(0.0)

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
			if (p - 6 - 2*e)<0.1:
				run=False

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




class NewPn5AAKWaveform(AAKWaveformBase, Pn5AAK):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):

        inspiral_kwargs["func"] = "pn5"

        AAKWaveformBase.__init__(
            self,
            PnTraj,  # trajectory class  EMRIInspiral
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )
