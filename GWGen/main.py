import os, psutil, shutil, gc, sys, time
import json
import argparse


"""
Redirect stdout
"""

orig_stdout = sys.stdout
stdout_file = orig_stdout #open(os.environ["HOME"]+"/WS_gwgen_output/debug/stdout.o", "w+")

print("Executing GWGen/main.py...",file=stdout_file)
print("{0}".format(time.ctime(time.time())), file=stdout_file)

dense_printing = True

try:
    import mpi4py as m4p
    from mpi4py import MPI
    usingmpi=True
except (ImportError, ModuleNotFoundError) as e:
    usingmpi=False

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action='store_true',default=False)
parser.add_argument("-p", "--plot", action='store_true',default=False)
parser.add_argument("--mpi", action="store_true", default=False)
parser.add_argument("--mp", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--overwrite", action="store_true", default=False)
args=parser.parse_args()


os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen.WFGenerator import *
from GWGen.Utils import GetInitialP, BHSpinAlphaCutoff,cartesian_product
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
matplotlib.use("Agg")
from matplotlib import figure

import superrad
from superrad import ultralight_boson

from astropy.units import yr, s
SecPerYR = yr.to(s)

import multiprocess as mp

try:
    import cupy as cp
    usingcupy=True
except (ImportError, ModuleNotFoundError) as e:
    usingcupy=False

#data directory relative to local parent GWGen
DataDirectory = os.path.abspath(os.path.dirname(__file__)) + "/Data/"
#NCPUs = 3
#DataDirectory = "/remote/pi213f/fell/DataStore/ProcaAroundKerrGW/GWGenOutput/"
#NCPUs = 32
#DataDirectory=os.environ["HOME"]+"/WS_gwgen_output/"
NCPUs=mp.cpu_count()


#generate plots
PlotData = args.plot

#boson spin
spin=1

#parameters
x0=1. #Initial Inclincation
qS=np.pi/4 #Sky Location Polar Angle in solar system barycenter coordinate system
phiS=0. #Sky Location Azimuthal Angle in solar system barycenter coordinate system
qK=1e-6 #Initial BH Spin Polar Angle in solar system barycenter coordinate system
phiK=0. #Initial BH Spin Azimuthal Angle in solar system barycenter coordinate system
dist=1. #Distance to source (Mpc)
mich=False #assume LISA long baseline response approximation

T=5.8 #LISA data run is 5 years. We set the max time to be longer because the proca cloud extends the inspiral time
dt=15 #time resolution in seconds

use_gpu=args.gpu #if CUDA or cupy is installed, this flag sets GPU parallelization
usingcupy=use_gpu #master variable to set use of cupy
usingmultipool=args.mp
usingmpi=args.mpi #master variable to set use of MPI

overwriteexisting = args.overwrite #overwrite existing solutions


print("Executing GWGen/main.py...",file=stdout_file)
print("{0}".format(time.ctime(time.time())), file=stdout_file)
print("\tUsing GPU: {0}\n\tUsing Cupy: {1}\n\tUsing Multiprocessing: {2}\n\tUsing MPI: {3}\n\tOverwrite existing solutions: {4}".format(use_gpu, usingcupy, usingmultipool, usingmpi,overwriteexisting))


# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "npoints": 110,  # we want a densely sampled trajectory
    "max_init_len": int(1e4),
    "dense_output":True
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")





def process(BHMASS, BHSpin,PROCAMASS,e0, plot=False,alphauppercutoff=0.335, alphalowercutoff=0.02,SecondaryMass=10, DataDir = DataDirectory, OverwriteSolution=False, mpirank=0, solcounter = 1, nsols = 1):

    if usingmpi:
        print("\n\nprocess {2} on solution {0} out of {1}".format(solcounter,nsols, mpirank))
        prepend_print_string = "Process rank {0} says: ".format(mpirank)
    elif usingmultipool:
        prepend_print_string = "Process rank {0} says: ".format(mp.current_process().name)
    else:
        prepend_print_string = ""



    alphaval = alphavalue(BHMASS, PROCAMASS)
    #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
    if alphaval>alphauppercutoff and spin==1:
        if dense_printing:
            print(prepend_print_string+"Alpha value {0:0.4f} beyond range of available data. Allowed range [[{1},{2}]]".format(alphaval,alphalowercutoff, alphauppercutoff))
        return None
    if alphaval<alphalowercutoff and spin==1:
        if dense_printing:
            print(prepend_print_string+"Alpha value {0:0.4f} below range of available data. Allowed range [[{1},{2}]]".format(alphaval,alphalowercutoff, alphauppercutoff))
        return None

    p0 = GetInitialP(BHMASS, e0) #approximate coalescence after 5 years for undressed system
    basefilename = "SMBHMass{0}_SMBHSpin{6}_SecMass{1}_ProcaMass{2}_ProcaSpin{3}_e0{4}_p0{5}.json".format(int(BHMASS),SecondaryMass,PROCAMASS,spin,int(e0*10)/10,int(p0*10)/10, BHSpin)
    filename = DataDir + "Output/"+basefilename

    #sanity check on initial parameters
    if p0<(get_separatrix(BHSpin, e0, 1.)+0.2):
        if dense_printing:
            print("Bad initial data: initial semi-latus rectum within 0.2 gravitational radii of separatrix! Skipping loop")
        return None

    if os.path.exists(filename):
        if OverwriteSolution:
            print(prepend_print_string+"Solution exists. Overwriting...")
        elif not OverwriteSolution:
            if dense_printing:
                print(prepend_print_string+"Solution already exists. Skipping...")
            return None


    print(prepend_print_string+"\nAlpha Value: {2}\nSMBH Mass: {0}\nProca Mass: {1}\nSMBH Spin: {5}\nEccentricity: {3}\nSemi-latus Rectum: {4}".format(BHMASS, PROCAMASS,alphaval, e0, p0, BHSpin), file=stdout_file)

    #Important: only pass copied version of kwargs as class can overwrite global variables. Should fix this....
    unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)
    moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)

    print(prepend_print_string+"Generating waveforms...", file=stdout_file)

    unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)
    unmoddedtraj = unmoddedwvcl.Trajectory
    print("\t Unmodded waveform generated. Generating modded waveform.")
    moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich,dt=dt, BosonSpin=spin, UltralightBoson = ulb)
    moddedtraj = moddedwvcl.Trajectory
    print(prepend_print_string+"Waveforms generated. Calculating figures of merit.", file=stdout_file)
    #azimuthal phase difference
    unmoddedphase = unmoddedtraj["Phi_phi"]
    moddedphase = moddedtraj["Phi_phi"]
    totalphasedifference = moddedphase[-1]-unmoddedphase[-1]
    totalorbitsdifference = totalphasedifference/(4*np.pi)

    ####Mismatch
    print(prepend_print_string+"Calculating mismatch", file=stdout_file)
    #truncate waveforms to be same length
    minlen = min([len(unmoddedwv), len(moddedwv)])
    unmoddedwv = unmoddedwv[:minlen]
    moddedwv = moddedwv[:minlen]
    if usingcupy:
        unmoddedwv = unmoddedwv.get()
        moddedwv = moddedwv.get()

    mismatch = get_mismatch(unmoddedwv, moddedwv,use_gpu=False)
    print(prepend_print_string+"Mismatch = {0}".format(mismatch), file=stdout_file)
    ####Faithfulness
    time = np.arange(minlen)*dt
    faith = Faithfulness(time, moddedwv, unmoddedwv,use_gpu=False)
    snr2 = WaveformInnerProduct(time, moddedwv, unmoddedwv, use_gpu=False)

    #data structure
    data = {
            "SMBHMASS": BHMASS,
            "SecondaryMass":SecondaryMass,
            "PROCAMASS":PROCAMASS,
            "p0":p0,
            "e0":e0,
            "BHSpin":BHSpin,
            "Trajectory Exit Reason": moddedwvcl.inspiral_generator.exit_reason,
            "mismatch":mismatch,
            "faithfulness":faith,
            "snr2":snr2,
            "DeltaNOrbits":totalorbitsdifference
            }




    #output data to disk
    jsondata = json.dumps(data)
    print(prepend_print_string+"Outputting data to: {0}".format(filename), file=stdout_file)
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
        alpha = {7}
        """.format(mismatch, BHMASS, PROCAMASS, spin,BHSpin, p0, e0, alphaval)
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
        print(prepend_print_string+"Saving plot to: {0}".format(DataDir+"Plots/"+basefilename[:-4]+"png"), file=stdout_file)
        fig.savefig(DataDir+"Plots/"+basefilename[:-4]+"png",dpi=300)
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

    del unmoddedwvcl,moddedwvcl,moddedwv,moddedtraj,unmoddedphase,moddedphase,totalphasedifference,totalorbitsdifference,minlen,mismatch,time,faith,data,jsondata
    if usingcupy:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return None



if __name__=='__main__':
    #run analysis

    DataDir = DataDirectory

    tmparr = np.linspace(1,9,9,dtype=np.int64) #strange floating point error when doing just np.arange(1,10,0.1) for np.linspace(1,10,91). Causes issues when saving numbers to filenames
    tmparr1 = np.linspace(1,9,81, dtype=np.float64)
    SMBHMasses = sorted([int(i) for i in np.kron(tmparr,[1e5, 1e6,1e7])]) #solar masses
    SMBHSpins = [int(100*i)/100 for i in np.linspace(0.6,0.9,10)]
    SecondaryMass = 10 #solar masses
    e0list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    ProcaMasses = [round(i,22) for i in np.kron(tmparr1, [1e-16,1e-17,1e-18,1e-19])] #eV   #again avoiding floating point errors

    #make sure output directory tree is built
    if not os.path.exists(DataDir+"Plots/"):
        os.mkdir(DataDir+"Plots/")

    if not os.path.exists(DataDir+"Output/"):
        os.mkdir(DataDir+"Output/")

    if not os.path.exists(DataDir+"debug/"):
        os.mkdir(DataDir+"debug/")

    if not usingmultipool and not usingmpi:
        PrettyPrint("Executing parallelized computation on {2} CPUs... \n\t Output Directory: {0}\n\t Plot Directory: {1}".format(DataDir+"Output/", DataDir+"Plot/", NCPUs))
        counter=1
        for bhmass in SMBHMasses:
            for pmass in ProcaMasses:
                for ecc in e0list:
                    for bhspin in SMBHSpins:
                        print("On iteration {0} out of {1}".format(counter, len(SMBHMasses)*len(ProcaMasses)*len(e0list)*len(SMBHSpins)))
                        process(bhmass, bhspin,pmass,ecc, plot=PlotData, SecondaryMass=SecondaryMass, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(bhspin),OverwriteSolution=overwriteexisting)
                        counter+=1


    if usingmultipool:
        parallel_func = lambda bhm, bhs, pmass, ecc: process(bhm, bhs, pmass, ecc, SecondaryMass=SecondaryMass, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(bhs),OverwriteSolution=overwriteexisting)
        parallel_args = cartesian_product(np.array(SMBHMasses),np.array(SMBHSpins), np.array(ProcaMasses), np.array(e0list))

        chunk_size = 20

        PrettyPrint("Executing parallelized computation on {2} CPUs... \n\t Output Directory: {0}\n\t Plot Directory: {1}".format(DataDir+"Output/", DataDir+"Plot/", NCPUs))
        starttime=time.time()
        with mp.Pool(processes=NCPUs) as poo:
            poo.starmap(parallel_func, parallel_args,chunksize=chunk_size)
        processtime = time.time()-starttime
        PrettyPrint("Time to complete computation: {0}".format(processtime))


    if usingmpi:
        comm = m4p.MPI.COMM_WORLD
        rank = comm.Get_rank()

        parallel_args = cartesian_product(np.array(SMBHMasses),np.array(SMBHSpins), np.array(ProcaMasses), np.array(e0list))
        parallel_func = lambda args,solcount,nsols: process(args[0], args[1], args[2], args[3], SecondaryMass=SecondaryMass, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(args[1]),mpirank=rank, solcounter=solcount,nsols=nsols,OverwriteSolution=overwriteexisting)

        def split(a, n):
            k, m = divmod(len(a), n)
            return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

        split_parallel_args = split(parallel_args, comm.Get_size())
        parallel_args_for_subprocesses = comm.scatter(split_parallel_args,root=0)

        if rank==0:
            print("Size of parameter space: {0}\nNumber MPI subprocesses: {1}".format(len(parallel_args), comm.Get_size()), file=stdout_file)
            print("shape of partitioned parameter space: {0}".format(np.shape(split_parallel_args)), file=stdout_file)

        with open("Rank{0}ProcessArguments.dat".format(rank), "w+") as file:
            for inx, val in enumerate(parallel_args_for_subprocesses):
                file.write("inx: {0}     val: {1}\n".format(inx+1,val))
        #main calculation
        counter = 1
        for inx, arg in enumerate(parallel_args_for_subprocesses):
            parallel_func(arg,counter,len(parallel_args_for_subprocesses))
            counter+=1

    sys.stdout=orig_stdout
    stdout_file.close()
