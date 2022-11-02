from mpmath import *
from scipy.integrate import DOP853
import numpy as np

from few.utils.baseclasses import TrajectoryBase
from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
from few.waveform import AAKWaveformBase
from few.summation.aakwave import AAKSummation

#hard values
####### make user inputs
a=0.2


def Power(x,n):
	return x**n

def Sqrt(x):
	return x**(1/2)
def Sign(x):
	return x/abs(x)

class PN:
	def __init__(self, epsilon): #epsilon is mass ratio
		self.epsilon=epsilon

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

		#define velocity
		v = 1/(p**(1/2))

		#setup guard for bad integration steps
		if e>=1.0 or e<1e-2 or p<6.0 or (p - 6 - 2* e) < 0.1:
			return [0.0, 0.0,0.0,0.0]

		# Azimuthal Frequency. Calculated using KerrGeodesics package from BlackHolePerturbationToolkit, KerrGeoFrequencies[a, p, 0, 1]
		Omega_phi = 1 / (a + p**(3/2))

		# Radial Frequency. Calculated using KerrGeodesics package from BlackHolePerturbationToolkit, KerrGeoFrequencies[a, p, 0, 1]
		Omega_r = (Sqrt(2*a + (-3 + p)*Sqrt(p))*Sqrt((-2*Power(a,2) + 6*a*Sqrt(p) + (-5 + p)*p +  Power(a - Sqrt(p),2)*Sign(Power(a,2) - 4*a*Sqrt(p) - (-4 + p)*p))/(2*a*Sqrt(p) + (-3 + p)*p)))/(Power(p,0.75)*(a + Power(p,1.5)))

		Edotcorr = 1.0  ############# CHANGE TO CORRECT PN VALUE
		EdotN = -32./5. * 1/(p**2) * v**6

		#include mass ratio
		Edot = epsilon*EdotN*Edotcorr
		Ldot = Edot/Omega_phi ########### Check this expression. Note it holds for circular orbits

		dEdp = (-3*Power(a,2) + 8*a*Sqrt(p) + (-6 + p)*p)/(2.*Power(2*a + (-3 + p)*Sqrt(p),1.5)*Power(p,1.75)) ########### Input correct values
		dLdp = -0.5*((3*Power(a,2) - 8*a*Sqrt(p) - (-6 + p)*p)*(a + Power(p,1.5)))/(Power(2*a + (-3 + p)*Sqrt(p),1.5)*Power(p,1.75)) ########### Input correct values

		pdot = Edot/(dEdp) + Ldot/(dLdp)

		edot = 0.0 #Circular orbit

		Phi_phi_dot = Omega_phi

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
