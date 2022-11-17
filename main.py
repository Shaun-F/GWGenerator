import GWGen
from GWGen.WFGenerator import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True



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

use_gpu=False


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

    ALPHACUTOFF = 0.02 #cutoff for dimensionless gravitational coupling. values larger than this correspond to proca clouds whose GW fluxes approximately exceed that of the EMRI

    for inx, BHMASS in enumerate(SMBHMasses):
        for inx2,PROCAMASS in enumerate(ProcaMasses):
            #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
            if alphavalue(BHMASS,PROCAMASS)>0.04:
                print("alpha>0.4. skipping loop")
                continue
            print("SMBH Mass: {0}\nProca Mass: {1}".format(BHMASS, PROCAMASS))
            unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
            unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
            unmoddedtraj = unmoddedwvcl.Trajectory

            moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
            moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt)
            moddedtraj = moddedwvcl.Trajectory

            #azimuthal phase difference
            unmoddedphase = unmoddedtraj["Phi_phi"]
            moddedphase = moddedtraj["Phi_phi"]
            totalphasedifference = moddedphase[-1]-unmoddedphase[-1]

            #Mismatch
            mismatch = get_mismatch(unmoddedwv, moddedwv)

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

            smallestwv = min([len(moddedwv), len(unmoddedwv)])
            dom3 = np.arange(smallestwv)*dt
            ax[1,0].plot(dom3, (moddedwv.real[0:smallestwv-1] - unmoddedwv.real[0:smallestwv-1]), label=r"h_{+}")
            ax[1,0].set_title("difference between with and with proca")
            ax[1,0].legend()
            ticks = ax[1,0].get_xticks()[1:-1];
            newlabs = [int(i)/100 for i in (ticks*100/(60*60*24*365))];
            ax[1,0].set_xticks(ticks, newlabs);
            ax[1,0].set_xlabel("years")
            ax[1,0].set_ylabel("strain")

            ax[1,1].axis('off')
            prop = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax[1,1].text(0.5,0.5, "mismatch = {0:.4f}".format(mismatch), bbox=prop, fontsize=14, verticalalignment='center', horizontalalignment='center')

            fig.savefig("data/unmoddedwvSMBHMass{0}SecMass{1}ProcMass{2}p0{3}e0{4}.png".format(BHMASS,SecondaryMass,PROCAMASS,p0,e0),dpi=300)
            plt.clf()
