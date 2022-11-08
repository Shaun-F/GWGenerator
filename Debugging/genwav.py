from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
print(path)
import sys
import GWGen
from GWGen.WFGenerator import *



import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
from few.utils.utility import omp_set_num_threads


class NewPn5AAKWaveform(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):

        inspiral_kwargs["func"] = "pn5"

        AAKWaveformBase.__init__(
            self,
            PnTraj,  # trajectory class
            AAKSummation, #Summation scheme for combining amplitude and phase information
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )

# set initial parameters
M = 1e6 #solar masses
mu = 1e0 #solar masses
a = 0.2 #dimensionless spin of SMBH
p0 = 14.0 #initial semi-major axis
e0 = 1e-6 #initial eccentricity
iota0 = 0 #inclination angle
Y0 = np.cos(iota0)
Phi_phi0 = 0. #inital azimuthal phase
Phi_theta0 = np.pi/2 #initial polar phase
Phi_r0 = 0. #initial radial phase


qS = 0.2 #sky location polar angle
phiS = 0.2 #sky location azimuthal angle
qK = 0.8  #SMBH Spin polar angle
phiK = 0.8 #SMBH spin azimuthal angle
dist = 1.0 #Distance in Gpc
mich = False #Return hplus and hcross
dt = 15.0 #time resolution in seconds
T = 0.001 #total observation time in years

use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

numthreads=16;
omp_set_num_threads(numthreads)
wave_generator = NewPn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False, num_threads=numthreads)
AAK_out = wave_generator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)

print(wave_generator.attributes_Pn5AAK())

time = np.arange(len(AAK_out.real))*dt
plt.plot(time, AAK_out.real**2 + AAK_out.imag**2)

plt.show()
