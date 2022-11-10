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
mu = 1e0
a = .5
p0 = 14.0
e0 = 0.3
iota0 = 0.
Y0 = np.cos(iota0)
T = 2.0
Phi_phi0 = 0.
Phi_theta0 = 0.0
Phi_r0 = 0.0





########### Few Trajectory###########
# initialize trajectory class
fewtraj = EMRIInspiral(func="pn5", enforce_schwarz_sep=False)

# run trajectory
tf, pf, ef, Yf, Phi_phif, Phi_rf, Phi_thetaf = fewtraj(M, mu, a, p0, e0, Y0, T=T)



########### My Trajectory ###########
traj = PNTraj(bhspin=1e-2, DeltaEFlux=-8e-12)
t,p,e,x,phiphi,phitheta,phir = traj(M,mu,a, p0,e0,Y0,T=T,npoints=100)


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
