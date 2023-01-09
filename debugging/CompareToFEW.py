import sys
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compare", help="what to compare", default="Waveform")
parser.add_argument("-g", "--gpu", action='store_true',default=False)
args=parser.parse_args()

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
a = .5
p0 = 10.0
e0 = 0.1
iota0 = 0.
Y0 = np.cos(iota0)
Phi_phi0 = 0.
Phi_theta0 = 0.0
Phi_r0 = 0.0


qS = 1e-20 #sky location polar angle
phiS = 0. #sky location azimuthal angle
qK = 0. #inital BH spin polar angle
phiK = 0. #initial BH spin azimuthal angle
dist = 1.0
mich = False
dt = 15
T = 5

if args.gpu:
    use_gpu = True
else:
    use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiralkwargs = {
    #"DENSE_STEPPING": 0,
    #"max_init_len": int(1e3),
    "npoints":100,
    "dense_output":False
}

# keyword arguments for summation generator (AAKSummation)
sumkwargs = {
    "use_gpu": use_gpu,
    "pad_output": False,
}

if args.compare=='Trajectory':
    ########### Few Trajectory###########
    # initialize trajectory class
    print("*********************** Generating FEW trajectory **************************")
    aa=time.time()
    fewtraj = EMRIInspiral(func="pn5", enforce_schwarz_sep=False)

    # run trajectory
    assert fewtraj.__module__ == 'few.trajectory.inspiral'
    tf, pf, ef, Yf, Phi_phif, Phi_rf, Phi_thetaf = fewtraj(M, mu, a, p0, e0, Y0, T=T)
    bb=time.time()
    print("time to generate FEW trajectory: {}".format(bb-aa))


    ########### My Trajectory ###########
    print("*********************** Generating my trajectory **************************")
    traj = EMRIWaveform()
    assert traj.__module__ == 'GWGen.WFGenerator.Kludge'
    traj.dense_output=True
    t,p,e,x,phiphi,phitheta,phir = traj.inspiral_generator(M,mu,a, p0,e0,Y0,T=T)
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
    axes[0,0].text(t[-1]/5, p[0]/2, "My final p: {0}\nFew Final p: {1}".format(p[-1], pf[-1]))
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

if args.compare=="Waveform":

    ############### FEW Waveform with AAK ###############
    print("*********************** Generating FEW waveform **************************")
    aa = time.time()
    wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiralkwargs.copy(), sum_kwargs=sumkwargs.copy(), use_gpu=use_gpu)
    print("SMBHMass {0} BHSpin {1} SecMass {2} qS {3} phiS {4} qK {5} phiK {6} dist {7}".format(M, a, mu, qS, phiS, qK, phiK, dist))

    FEWwaveform = wave_generator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
    bb=time.time()
    print("time to generate FEW waveform: {}".format(bb-aa))

    ############### My Waveform ###############

    print("*********************** Generating my waveform **************************")
    wfgenerator = EMRIWaveform(inspiral_kwargs=inspiralkwargs.copy(), sum_kwargs=sumkwargs.copy(), use_gpu=use_gpu)
    mywf =        wfgenerator(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
    cc=time.time()
    print("time to generate my waveform: {}".format(cc-bb))



    ############### plot #################
    FEWwaveform = FEWwaveform.get()
    mywf = mywf.get()
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
    startinginx = 0
    cutoffinx = 2000;
    axes[1,0].plot(t[startinginx:cutoffinx], mywf.real[startinginx:cutoffinx]);
    axes[1,0].plot(tf[startinginx:cutoffinx], FEWwaveform.real[startinginx:cutoffinx]);
    xticks = axes[1,0].get_xticks()[1:-1]
    axes[1,0].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*31))]);
    axes[1,0].set_xlabel("months");
    axes[1,0].set_ylabel("strain");

    axes[1,1].text(0.5,0.5, "Mismatch = {0}".format(get_mismatch(FEWwaveform, mywf)))
    axes[1,1].axis("off")

    plt.show()
