import os, psutil, shutil, gc, sys
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action='store_true',default=False)
parser.add_argument("-p", "--plot", action='store_true',default=False)
args=parser.parse_args()


os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen.WFGenerator import *
from GWGen.Utils import GetInitialP
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
matplotlib.use("Agg")
from matplotlib import figure

import superrad
from superrad import ultralight_boson

from astropy.units import yr, s
SecPerYR = yr.to(s)

#data directory relative to local parent GWGen
DataDirectory = os.path.abspath(os.path.dirname(__file__)) + "/Data/"
#DataDirectory = "/remote/pi213f/fell/DataStore/ProcaAroundKerrGW/GWGenOutput/"

#generate plots
PlotData = True

#boson spin
spin=1

#parameters
BHSpin=0.9 #SMBH Spin
x0=1. #Initial Inclincation
qS=np.pi/4 #Sky Location Polar Angle in solar system barycenter coordinate system
phiS=0. #Sky Location Azimuthal Angle in solar system barycenter coordinate system
qK=1e-6 #Initial BH Spin Polar Angle in solar system barycenter coordinate system
phiK=0. #Initial BH Spin Azimuthal Angle in solar system barycenter coordinate system
dist=1. #Distance to source (Mpc)
mich=False #assume LISA long baseline response approximation

T=8 #LISA data run is 5 years. We set the max time to be longer because the proca cloud extends the inspiral time
dt=15 #time resolution in seconds

use_gpu=False #if CUDA or cupy is installed, this flag sets GPU parallelization


# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "npoints": 130,  # we want a densely sampled trajectory
    "max_init_len": int(1e4),
    "dense_output":True
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")





def process(BHMASS, PROCAMASS,e0, plot=False,alphauppercutoff=0.335, alphalowercutoff=0.06,SecondaryMass=10, DataDir = DataDirectory, OverwriteSolution=False):

    alphaval = alphavalue(BHMASS, PROCAMASS)
    #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
    if alphaval>alphauppercutoff and spin==1:
        return None
    if alphaval<alphalowercutoff and spin==1:
        return None


    filename = DataDir + "Output/SMBHMass{0}_SecMass{1}_ProcaMass{2}_ProcaSpin{3}_e0{4}_p0{5}.json".format(int(BHMASS),SecondaryMass,PROCAMASS,spin,int(e0*10)/10,int(p0*10)/10)

    if os.path.exists(filename):
        return None


    p0 = GetInitialP(BHMASS, e0) #approximate coalescence after 5 years for undressed system
    print("Alpha Value: {2}\nSMBH Mass: {0}\nProca Mass: {1}\n Eccentricity: {3}\nSemi-latus Rectum: {4}".format(BHMASS, PROCAMASS,alphaval, e0, p0))

    #Important: only pass copied version of kwargs as class can overwrite global variables. Should fix this....
    unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)
    moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)

    unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
    unmoddedtraj = unmoddedwvcl.Trajectory

    moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt, BosonSpin=spin, UltralightBoson = ulb)
    moddedtraj = moddedwvcl.Trajectory

    #azimuthal phase difference
    unmoddedphase = unmoddedtraj["Phi_phi"]
    moddedphase = moddedtraj["Phi_phi"]
    totalphasedifference = moddedphase[-1]-unmoddedphase[-1]
    totalorbitsdifference = totalphasedifference/(4*np.pi)

    ####Mismatch
    #truncate waveforms to be same length
    minlen = min([len(unmoddedwv), len(moddedwv)])
    unmoddedwv = unmoddedwv[:minlen]
    moddedwv = moddedwv[:minlen]
    mismatch = get_mismatch(unmoddedwv, moddedwv)

    ####Faithfulness
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
            "Trajectory Exit Reason": moddedwvcl.inspiral_generator.exit_reason,
            "mismatch":mismatch,
            "faithfulness":faith,
            "DeltaNOrbits":totalorbitsdifference
            }


    #output data to disk
    jsondata = json.dumps(data)
    with open(filename, "w") as file:
        file.write(jsondata)

    if plot:
        #plots
        fig = figure.Figure(figsize=(16,8))
        ax = fig.subplots(3,2)
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
        string = r"""mismatch = {0:.4f}
        SMBHMass = {1}
        Proca Mass = {2}
        BosonSpin = {3}
        BHSpin = {4}
        p0 = {5}
        e0 = {6}
        """.format(mismatch, BHMASS, PROCAMASS, spin,BHSpin, p0, e0)
        ax[1,1].text(0.5,0.5, string, bbox=prop, fontsize=14, verticalalignment='center', horizontalalignment='center')

        ax[2,0].plot(moddedtraj["t"]/SecPerYR, moddedtraj["p"], label="With Proca")
        ax[2,0].plot(unmoddedtraj["t"]/SecPerYR, unmoddedtraj["p"], label="Without Proca")
        ax[2,0].set_title("Semi-latus Rectum Evolution")
        ax[2,0].set_xlabel("time (yr)")
        ax[2,0].set_ylabel("semi-latus rectum")
        ax[2,0].legend()

        ax[2,1].plot(moddedtraj["t"]/SecPerYR, moddedtraj["e"], label="With Proca")
        ax[2,1].plot(unmoddedtraj["t"]/SecPerYR, unmoddedtraj["e"], label="Without Proca")
        ax[2,1].set_title("Eccentricity Evolution")
        ax[2,1].set_xlabel("time (yr)")
        ax[2,1].set_ylabel("eccentricity")
        ax[2,1].legend()

        fig.savefig(DataDir+"Plots/"+"Plot_SMBHMass{0}_SecMass{1}_ProcMass{2}_ProcSpin{5}_p0{3}_e0{4}.png".format(BHMASS,SecondaryMass,PROCAMASS,p0,e0,spin),dpi=300)
        #strange memory leak in savefig method. Calling different clear functions and using different Figure instance resolves problem
        plt.close(fig)
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()

    if args.debug:
        debugdir = DataDir+"debug/";
        proc = psutil.Process(os.getpid())
        meminuse = proc.memory_info().rss/(1024**2)
        with open(debugdir+"memuse.txt","a+") as file:
            file.write("\n")
            file.write(str(meminuse))

    return None



if __name__=='__main__':
    #run analysis

    DataDir = DataDirectory

    tmparr = np.linspace(1,9,9,dtype=np.int64) #strange floating point error when doing just np.arange(1,10,0.1) for np.linspace(1,10,91). Causes issues when saving numbers to filenames
    SMBHMasses = sorted([int(i) for i in np.kron(tmparr,[1e5, 1e6,1e7])]) #solar masses
    SecondaryMass = 10 #solar masses
    e0list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    ProcaMasses = [round(i,22) for i in np.kron(tmparr, [1e-16,1e-17,1e-18,1e-19])] #eV   #again avoiding floating point errors

    #make sure output directory tree is built
    if not os.path.exists(DataDir+"Plots/"):
        os.mkdir(DataDir+"Plots/")

    if not os.path.exists(DataDir+"Output/"):
        os.mkdir(DataDir+"Output/")

    if os.path.exists(DataDir+"debug/"):
        shutil.rmtree(DataDir+"debug/")
    os.mkdir(DataDir+"debug/")

    for bhmass in SMBHMasses:
        for pmass in ProcaMasses:
            for ecc in e0list:
                process(bhmass, pmass,ecc, plot=PlotData, SecondaryMass=SecondaryMass, DataDir=DataDir)






















"""
alpha values for mode = 1 overtone = 0
[0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335]

"""
