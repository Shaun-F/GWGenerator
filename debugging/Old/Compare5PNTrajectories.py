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
M = 1e6
mu = 1e1
a = .5
p0 = 10.0
e0 = 0.3
iota0 = 0.
Y0 = np.cos(iota0)
T = 10.0
Phi_phi0 = 0.
Phi_theta0 = 0.0
Phi_r0 = 0.0

use_gpu = True

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    #"DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    #"max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "npoints":50,
    "dense_output":False
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}




########### Few Trajectory###########
# initialize trajectory class
print("*********************** Generating FEW trajectory **************************")
aa=time.time()
fewtraj = EMRIInspiral(func="pn5", enforce_schwarz_sep=False)

# run trajectory
tf, pf, ef, Yf, Phi_phif, Phi_rf, Phi_thetaf = fewtraj(M, mu, a, p0, e0, Y0, T=T)
bb=time.time()
print("time to generate FEW trajectory: {}".format(bb-aa))


########### My Trajectory ###########
print("*********************** Generating my trajectory **************************")
traj = EMRIWaveform()
traj.dense_output=True
t,p,e,x,phiphi,phitheta,phir = traj.inspiral_generator(M,mu,a, p0,e0,Y0,T=T,npoints=10)
cc=time.time()
print("time to generate my waveform: {}".format(cc-bb))
print("Size of trajectory {0}".format(len(t)))

fig,axes=plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5)
fig.set_size_inches(16,8)
axes[0,0].set_title("My trajectory")
axes[0,0].set_ylabel("p")
axes[0,0].set_xlabel("t")
axes[0,0].plot(tf,pf, label="FEW")
axes[0,0].plot(t,p, label="Mine")
axes[0,0].legend()

axes[0,1].set_title("My trajectory")
axes[0,1].set_ylabel("e")
axes[0,1].set_xlabel("t")
axes[0,1].plot(tf,ef, label="FEW")
axes[0,1].plot(t,e, label="Mine")
axes[0,1].legend()

axes[1,0].set_title("My trajectory")
axes[1,0].set_ylabel("e")
axes[1,0].set_xlabel("p")
axes[1,0].plot(pf,ef, label="FEW")
axes[1,0].plot(p,e, label="Mine")
axes[1,0].legend()

axes[1,1].set_title("My trajectory")
axes[1,1].set_ylabel("phi phi")
axes[1,1].set_xlabel("t")
axes[1,1].plot(tf,Phi_phif, label="FEW")
axes[1,1].plot(t,phiphi, label="Mine")
axes[1,1].legend()

plt.show()
