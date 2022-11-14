import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
from few.utils.utility import *

os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen.WFGenerator import *

# set initial parameters
M = 1e5
mu = 1e1
a = 0.9
p0 = 10.0
e0 = 1e-6
iota0 = 0.0
Y0 = np.cos(iota0)
Phi_phi0 = 0.
Phi_theta0 =0.
Phi_r0 = 0.


qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False
dt = 15.0
T = 2.0

use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "DeltaEFlux":0
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}





############### FEW Waveform with AAK ###############


#wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
#FEWwaveform = wave_generator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)


############### My Waveform ###############

wfgenerator = NewPN5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)

mywf = wfgenerator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)



############### plot #################
fig,axes = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)
fig.set_size_inches(16,8)


t = np.arange(len(mywf)) * dt
axes[0,0].set_title("My Model")
axes[0,0].plot(t, mywf.real);
xticks = axes[0,0].get_xticks()[1:-1]
axes[0,0].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*365))]);
axes[0,0].set_xlabel("years");
axes[0,0].set_ylabel("strain");

"""
tf =np.arange(len(FEWwaveform)) * dt;
axes[0,1].set_title("FEW Model")
axes[0,1].plot(tf, FEWwaveform.real);
xticks = axes[0,1].get_xticks()[1:-1]
axes[0,1].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*365))]);
axes[0,1].set_xlabel("years");
axes[0,1].set_ylabel("strain");

axes[1,0].set_title("Overlap")
startinginx = 500000
cutoffinx = 501000;
axes[1,0].plot(t[startinginx:cutoffinx], mywf.real[startinginx:cutoffinx]);
axes[1,0].plot(tf[startinginx:cutoffinx], FEWwaveform.real[startinginx:cutoffinx]);
xticks = axes[1,0].get_xticks()[1:-1]
axes[1,0].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*31))]);
axes[1,0].set_xlabel("months");
axes[1,0].set_ylabel("strain");

axes[1,1].text(0.5,0.5, "Mismatch = {0}".format(get_mismatch(FEWwaveform, mywf)))
axes[1,1].axis("off")
"""
plt.show()
