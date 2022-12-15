import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.fft

import GWGen
from GWGen.WFGenerator import *

# set initial parameters
M = 1e6
m = 1e1
mu = 4e-17
a = 0.6
p0 = 10.0
e0 = 0.2
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
T = 2

alphaval = alphavalue(M,mu)
print(r"alpha = {0}".format(alphaval))


use_gpu = True

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
insp_kwargs = {
    "npoints": 100,  # we want a densely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

########## without proca #############

ProcaInspiralKwargs = insp_kwargs.copy()
ProcaSumKwargs = sum_kwargs.copy()
withprocagen = EMRIWithProcaWaveform(inspiral_kwargs=ProcaInspiralKwargs,sum_kwargs=ProcaSumKwargs)
withproca = withprocagen(M,m,mu,a,p0,e0,Y0,T=T,qS=qS,phiS=phiS,qK=qK,phiK=phiK,dist=dist,mich=mich)




############# with proca ##############

WithoutProcaInspiralKwargs = insp_kwargs.copy()
WithoutProcaSumKwargs=sum_kwargs.copy()
NewMass = withprocagen.FinalBHMass
NewSpin = withprocagen.FinalBHSpin
withoutprocagen = EMRIWaveform(inspiral_kwargs=WithoutProcaInspiralKwargs, sum_kwargs=WithoutProcaSumKwargs, use_gpu=False)
withoutproca = withoutprocagen(NewMass, m, NewSpin, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)








titlefontsize=10
legendsize=6;
if use_gpu:
    withoutproca = withoutproca.get()
    withproca=withproca.get()
mismatch = get_mismatch(withoutproca, withproca)

tp = np.arange(len(withproca)) * dt;
twp = np.arange(len(withoutproca)) * dt

fig,ax = plt.subplots(5,2, figsize=(45,200))
plt.subplots_adjust(hspace=2, wspace=0.5)
ax[0,0].plot(tp, withproca.real)
ax[0,0].set_title("With proca", fontsize=titlefontsize)
ticks = ax[0,0].get_xticks()[1:-1];
newlabs = np.array([int(i)/100 for i in (ticks*100/(60*60*24*365))]);
ax[0,0].set_xticks(ticks, newlabs);
ax[0,0].set_xlabel("years");
ax[0,0].set_ylabel("strain");

ax[0,1].plot(twp, withoutproca.real)
ax[0,1].set_title("Without proca", fontsize=titlefontsize)
ticks = ax[0,1].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[0,1].set_xticks(ticks, newlabs);
ax[0,1].set_xlabel("years");
ax[0,1].set_ylabel("strain");

minwave = min([len(withproca), len(withoutproca)])
dom =tp[0:minwave-1]
ax[1,0].plot(dom, (withproca.real[0:minwave-1]-withoutproca.real[0:minwave-1]));
ax[1,0].set_title("difference between with and without proca \n for alpha={0}. Mismatch={1}".format(alphaval,mismatch), fontdict={"fontsize":9}, fontsize=titlefontsize);
ticks = ax[1,0].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[1,0].set_xticks(ticks, newlabs);
ax[1,0].set_xlabel("years");
ax[1,0].set_ylabel("strain");


minsize = min([len(tp), len(twp)])
dom = withprocagen.Trajectory["t"]
#N_orbits = phase/pi
Difference = (withprocagen.Trajectory["Phi_phi"] - withoutprocagen.Trajectory["Phi_phi"])/np.pi
ax[1,1].plot(withprocagen.Trajectory["t"], Difference, label="withproca")
ax[1,1].set_title(r"$\Delta N$ number of GW cycles", fontsize=titlefontsize)
ax[1,1].legend(prop={'size':legendsize})
ticks = ax[1,1].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[1,1].set_xticks(ticks, newlabs);
ax[1,1].set_xlabel("years");
ax[1,1].set_ylabel(r"$\Delta N_{cycles}$");


ax[2,0].plot(withprocagen.Trajectory["t"], withprocagen.Trajectory["p"], label="with proca")
ax[2,0].set_title(" semi-latus rectum evolution", fontsize=titlefontsize)
ax[2,0].legend(prop={'size':legendsize})
ticks = ax[2,0].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[2,0].set_xticks(ticks, newlabs);
ax[2,0].set_xlabel("years");
ax[2,0].set_ylabel("p");

ax[2,1].plot(withoutprocagen.Trajectory["t"], withoutprocagen.Trajectory["p"], label="without proca")
ax[2,1].set_title(" semi-latus rectum evolution", fontsize=titlefontsize)
ax[2,1].legend(prop={'size':legendsize})
ticks = ax[2,1].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[2,1].set_xticks(ticks, newlabs);
ax[2,1].set_xlabel("years");
ax[2,1].set_ylabel("p");


minsize = min([len(tp), len(twp)])
dom = tp[0:minsize-1]
ax[3,0].plot(withprocagen.Trajectory["t"], withprocagen.Trajectory["p"], label="withproca")
ax[3,0].plot(withoutprocagen.Trajectory["t"], withoutprocagen.Trajectory["p"], label="withoutproca")
ax[3,0].set_title(" semi-latus rectum evolution", fontsize=titlefontsize)
ax[3,0].legend(prop={'size':legendsize})
ticks = ax[3,0].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[3,0].set_xticks(ticks, newlabs);
ax[3,0].set_xlabel("years");
ax[3,0].set_ylabel("p");


minsize = min([len(tp), len(twp)])
dom = tp[0:minsize-1]
ax[3,1].plot(withprocagen.Trajectory["p"], withprocagen.Trajectory["e"], label="withproca")
ax[3,1].plot(withoutprocagen.Trajectory["p"], withoutprocagen.Trajectory["e"], label="withoutproca")
ax[3,1].set_title(" configuration space trajectory", fontsize=titlefontsize)
ax[3,1].legend(prop={'size':legendsize})
ax[3,1].set_xlabel("e");
ax[3,1].set_ylabel("p");


minsize = min([len(tp), len(twp)])
dom = tp[0:minsize-1]
ax[4,0].plot(withprocagen.Trajectory["t"], withprocagen.Trajectory["e"], label="withproca")
ax[4,0].plot(withoutprocagen.Trajectory["t"], withoutprocagen.Trajectory["e"], label="withoutproca")
ax[4,0].set_title(" eccentricity evolution", fontsize=titlefontsize)
ax[4,0].legend(prop={'size':legendsize})
ticks = ax[4,0].get_xticks()[1:-1];
newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
ax[4,0].set_xticks(ticks, newlabs);
ax[4,0].set_xlabel("years");
ax[4,0].set_ylabel("e");

plt.show()
