import os
import sys
import json


os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen.WFGenerator import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import joblib
from joblib import Parallel, delayed
import superrad
from superrad import ultralight_boson

#number of cpus to use for parallelization
NCPUs = 4

#data directory relative to local parent GWGen
DataDirectory = os.path.abspath(os.path.dirname(__file__)) + "/Data/"
#DataDirectory = "/remote/pi213f/fell/DataStore/ProcaAroundKerrGW/GWGenOutput/"

#generate plots
PlotData = True

#boson spin
spin=1

#parameters
BHSpin=0.9 #SMBH Spin
p0=10. #Initial Semilatus Rectum
e0=0.2 #Initial Eccentricity
x0=1. #Initial Inclincation
qS=np.pi/4 #Sky Location Polar Angle
phiS=0. #Sky Location Azimuthal Angle
qK=1e-6 #Initial BH Spin Polar Angle. We want this to be as close to zero as allowed by FEW package. This must zero so the secondary BH orbits on equator of SMBH
phiK=0. #Initial BH Spin Azimuthal Angle
dist=1. #Distance to source (Mpc)
mich=False #assume LISA long baseline response approximation

T=5 #LISA data run
dt=15 #time resolution in seconds

use_gpu=False #if CUDA or cupy is installed, this flag sets GPU parallelization


# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "npoints": 100,  # we want a densely sampled trajectory
    "max_init_len": int(1e3),
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")


def process(BHMASS, PROCAMASS, plot=False,alphauppercutoff=0.335, alphalowercutoff=0.06,SecondaryMass=10, DataDir = DataDirectory):

    alphaval = alphavalue(BHMASS, PROCAMASS)
    #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
    if alphaval>alphauppercutoff and spin==1:
        return None
    if alphaval<alphalowercutoff and spin==1:
        return None

    print("Alpha Value: {2}\nSMBH Mass: {0}\nProca Mass: {1}".format(BHMASS, PROCAMASS,alphaval))

    #Important: only pass copied version of kwargs as class can overwrite global variables. Should fix this....
    unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)
    moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)

    unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
    unmoddedtraj = unmoddedwvcl.Trajectory

    moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt, BosonSpin=spin, UltralightBoson = ulb)
    moddedtraj = moddedwvcl.Trajectory





    return None



if __name__=='__main__':
    #run analysis

    tmparr = [int(i*10)/10 for i in np.arange(1,10,0.1)] #strange floating point error when doing just np.arange(1,10,0.1) for np.linspace(1,10,91). Causes issues when saving numbers to filenames
    SMBHMasses = [int(i) for i in np.kron(tmparr,[1e5, 1e6,1e7])] #solar masses
    SecondaryMass = 10 #solar masses
    ProcaMasses = [round(i,22) for i in np.kron(tmparr, [1e-16,1e-17,1e-18,1e-19])] #eV   #again avoiding floating point errors

    [process(bhmass, pmass,plot=PlotData, SecondaryMass=SecondaryMass) for bhmass in SMBHMasses for pmass in ProcaMasses]























"""
alpha values for mode = 1 overtone = 0
[0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335]

"""
