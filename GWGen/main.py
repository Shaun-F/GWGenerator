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


#data directory
DataDirectory = "/Data/"

#generate plots
PlotData = False

#boson spin
spin=1

#parameters
BHSpin=0.9
p0=10
e0=0.2
x0=1
qS=0.2
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
    "npoints": 50,  # we want a densely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

if __name__=='__main__':
    #run analysis

    tmparr = np.arange(1,10,1)
    SMBHMasses = np.kron(tmparr,[1e6,1e7]) #solar masses
    SecondaryMass = 10 #solar masses
    ProcaMasses = np.kron(tmparr, [1e-14,1e-15,1e-16,1e-17,1e-18,1e-19]) #eV

    PROCAALPHACUTOFF = 0.04 #cutoff for dimensionless gravitational coupling. values larger than this correspond to proca clouds whose GW fluxes approximately exceed that of the EMRI

    def process(BHMASS, PROCAMASS, plot=False):
        #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
        if alphavalue(BHMASS,PROCAMASS)>PROCAALPHACUTOFF and spin==1:
            print("alpha>0.4. skipping loop")
            return None

        print("SMBH Mass: {0}\nProca Mass: {1}".format(BHMASS, PROCAMASS))
        unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
        unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
        unmoddedtraj = unmoddedwvcl.Trajectory

        moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
        moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt, BosonSpin=spin)
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
                "SMBHMASS": SMBHMASS,
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
        with open(DataDirectory+"SMBHMass{0}_SecMass{1}_ProcMass{2}_ProcSpin{3}.png".format(BHMASS,SecondaryMass,PROCAMASS,spin), "w") as file:
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

    Parallel(n_jobs=8)(delayed(process)(bhmass, pmass,plot=PlotData) for bhmass in SMBHMasses for pmass in ProcaMasses)
