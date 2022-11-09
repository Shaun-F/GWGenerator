from scipy.integrate import DOP853
from mpmath import *
mp.dps=25
mp.pretty=True

import numpy as np
from GWGen.Utils import *
from GWGen.UndressedFluxes import *

from few.trajectory.inspiral import EMRIInspiral
from few.utils.baseclasses import TrajectoryBase
from few.utils.constants import MTSUN_SI, YRSID_SI, Pi
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
from few.utils.utility import *
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
		self.UndressedEFlux = lambda e,p: Analytic5PNEFlux(e,p,self.a)
		self.UndressedLFlux = lambda e,p: Analytic5PNLFlux(e,p,self.a)


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
		try:
			# Azimuthal Frequency
			Omega_phi = self.AzimuthalFrequency(ecc,semimaj);

			# Radial Frequency
			Omega_r = self.RadialFrequency(ecc,semimaj)

			#Energy flux
			EdotN = self.UndressedEFlux(ecc,semimaj)

			#Angular momentum
			LdotN = self.UndressedLFlux(ecc,semimaj)
		except TypeError:
			print("ERROR: type error in frequency and flux generation as (e,p)=({0},{1})".format(ecc,semimaj))

		#include mass ratio
		Edot = epsilon*EdotN
		Ldot = epsilon*LdotN

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
		self.PNEvaluator = PN(epsilon,bhspin=a)
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
			if (p - get_separatrix(a,e,1.))<0.1:
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




class NewPn5AAKWaveform(AAKWaveformBase, Pn5AAK):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):

        inspiral_kwargs["func"] = "pn5"

        AAKWaveformBase.__init__(
            self,
            PNTraj,  # trajectory class  EMRIInspiral
            AAKSummation, #Summation module for combining amplitude and phase information
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )








##########Delete the follow class

class testPnTrajectory(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, T=1.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M
        self.PNEvaluator=PN(epsilon,bhspin=a)
        integrator = DOP853(self.PNEvaluator, 0.0, y0, T, max_step = T/10)

        t_out, p_out, e_out = [0.], [p0], [e0]
        Phi_phi_out, Phi_r_out = [0.], [0.]

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:
            integrator.step()
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)

           #catch separatrix crossing and halt integration
            if (p - get_separatrix(a,e,1.))<0.1:
                run=False
                exit_reason="Passed separatrix"

            if e<0 or e>=1:
                run=False
                exit_reason="Ecccentricity exceeded bounds"

        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
