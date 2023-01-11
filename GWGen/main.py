import os
import sys
import json
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
NCPUs = 6

#data directory relative to local parent GWGen
DataDirectory = os.path.abspath(os.path.dirname(__file__)) + "/../Data/Output/"
#DataDirectory = "/remote/pi213f/fell/DataStore/ProcaAroundKerrGW/GWGenOutput/"

#generate plots
PlotData = False

#boson spin
spin=1

#parameters
BHSpin=0.9
p0=10.
e0=0.2
x0=1.
qS=1e-20
phiS=0.
qK=0.
phiK=0.
dist=1.
mich=False

T=5 #LISA data run
dt=15 #time resolution in seconds

use_gpu=False #if CUDA or cupy is installed, this flag sets GPU parallelization


# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "npoints": 99,  # we want a densely sampled trajectory
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


def process(BHMASS, PROCAMASS, plot=False,alphauppercutoff=0.335, alphalowercutoff=0.06,SecondaryMass=10):

    alphaval = alphavalue(BHMASS, PROCAMASS)
    print("Alpha Value: {0}".format(alphaval))
    #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
    if alphaval>alphauppercutoff and spin==1:
        print("alpha>{0}. skipping loop".format(alphauppercutoff))
        return None
    if alphaval<alphalowercutoff and spin==1:
        print("alpha<{0}. skipping loop".format(alphalowercutoff))
        return None

    print("SMBH Mass: {0}\nProca Mass: {1}".format(BHMASS, PROCAMASS))
    unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
    unmoddedtraj = unmoddedwvcl.Trajectory

    moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt, BosonSpin=spin, UltralightBoson = ulb)
    moddedtraj = moddedwvcl.Trajectory

    #azimuthal phase difference
    unmoddedphase = unmoddedtraj["Phi_phi"]
    moddedphase = moddedtraj["Phi_phi"]
    totalphasedifference = moddedphase[-1]-unmoddedphase[-1]

    #Mismatch
    mismatch = get_mismatch(unmoddedwv, moddedwv)

    #Faithfulness
    minlen = min([len(moddedwv), len(unmoddedwv)])
    time = np.arange(minlen)*dt
    faith = Faithfulness(time, moddedwv, unmoddedwv)

    #data structure
    data = {
            "SMBHMASS": BHMASS,
            "SecondaryMass":SecondaryMass,
            "PROCAMASS":PROCAMASS,
            "p0":p0,
            "e0":e0,
            "BHSpin":spin,
            "mismatch":mismatch,
            "faithfulness":faith
            }


    #output data to disk
    jsondata = json.dumps(data)
    filename = DataDirectory + "SMBHMass{0}_SecMass{1}_ProcMass{2}_ProcSpin{3}.json".format(BHMASS,SecondaryMass,PROCAMASS,spin)
    with open(filename, "w") as file:
        file.write(jsondata)



    if plot:
        #plots
        fig,ax = plt.subplots(2,2,figsize=(16,8))
        plt.subplots_adjust(wspace=0.5,hspace=0.5)

        dom1 = np.arange(len(unmoddedwv))*dt
        ax[0,0].plot(dom1, unmoddedwv.real, label=r"h_{+}")
        ax[0,0].set_title("Gravitational Waveform (without proca)")
        ax[0,0].legend()
        ticks = ax[0,0].get_xticks()[1:-1];
        newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
        ax[0,0].set_xticks(ticks, newlabs);

        dom2 = np.arange(len(moddedwv))*dt
        ax[0,1].plot(dom2, moddedwv.real, label=r"h_{+}")
        ax[0,1].set_title("Gravitational Waveform (with proca)")
        ax[0,1].legend()
        ticks = ax[0,1].get_xticks()[1:-1];
        newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
        ax[0,1].set_xticks(ticks, newlabs);
        ax[0,1].set_xlabel("years")
        ax[0,1].set_ylabel("strain")

        smallestwv = min([len(moddedwv), len(unmoddedwv)])-1
        dom3 = np.arange(smallestwv)*dt
        ax[1,0].plot(dom3, (moddedwv[0:smallestwv].real - unmoddedwv[0:smallestwv].real), label=r"h_{+}")
        ax[1,0].set_title("difference between with and with proca")
        ax[1,0].legend()
        ticks = ax[1,0].get_xticks()[1:-1];
        newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
        ax[1,0].set_xticks(ticks, newlabs);
        ax[1,0].set_xlabel("years")
        ax[1,0].set_ylabel("strain")

        ax[1,1].axis('off')
        prop = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        str = r"""mismatch = {0:.4f}
        SMBHMass = {1}
        Proca Mass = {2}
        BosonSpin = {3}
        BHSpin = {4}
        p0 = {5}
        e0 = {6}
        """.format(mismatch, BHMASS, PROCAMASS, spin,BHSpin, p0, e0)
        ax[1,1].text(0.5,0.5, str, bbox=prop, fontsize=14, verticalalignment='center', horizontalalignment='center')

        fig.savefig(DataDirectory+"Plot_SMBHMass{0}_SecMass{1}_ProcMass{2}_ProcSpin{5}_p0{3}_e0{4}.png".format(BHMASS,SecondaryMass,PROCAMASS,p0,e0,spin),dpi=300)
        plt.clf()




if __name__=='__main__':
    #run analysis

    tmparr = np.arange(1,10,.1)
    SMBHMasses = np.kron(tmparr,[1e5, 1e6,1e7]) #solar masses
    SecondaryMass = 10 #solar masses
    ProcaMasses = np.kron(tmparr, [1e-16,1e-17,1e-18,1e-19]) #eV

    Parallel(n_jobs=NCPUs, prefer="threads")(delayed(process)(bhmass, pmass,plot=PlotData, SecondaryMass=SecondaryMass) for bhmass in SMBHMasses for pmass in ProcaMasses)























"""
alpha values for mode = 1 overtone = 0
[0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335]

"""
