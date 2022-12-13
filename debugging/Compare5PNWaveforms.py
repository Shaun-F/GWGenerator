import sys
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
from few.utils.utility import *

import GWGen
from GWGen.WFGenerator import *

# set initial parameters
M = 1e6
mu = 1e1
a = .2
p0 = 10.0
e0 = 0.7
iota0 = 0.
Y0 = np.cos(iota0)
Phi_phi0 = 0.
Phi_theta0 = 0.
Phi_r0 = 0.


qS = np.pi/4 #sky location polar angle
phiS = 0. #sky location azimuthal angle
qK = 0. #inital BH spin polar angle
phiK = 0. #initial BH spin azimuthal angle
dist = 1.0
mich = False
dt = 15
T = 0.001

use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    #"DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    #"max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "npoints":11
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}





############### FEW Waveform with AAK ###############
print("*********************** Generating FEW waveform **************************")
aa = time.time()
wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
FEWwaveform = wave_generator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
bb=time.time()
print("time to generate FEW waveform: {}".format(bb-aa))

############### My Waveform ###############

print("*********************** Generating my waveform **************************")
wfgenerator = EMRIWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
mywf = wfgenerator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
cc=time.time()
print("time to generate my waveform: {}".format(cc-bb))



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

plt.show()
