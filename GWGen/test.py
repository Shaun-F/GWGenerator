from main import *
from mpi4py import MPI


smbhmass = 100000
smbhspin = 0.9
p0 = 34.8
e0 = 0.3
pmass = 1e-16

"""
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
"""

DataDir = os.path.abspath(os.path.dirname(__file__)) + "/Data/"

#parallel_func = lambda args,solcount,nsols: process(args[0], args[1], args[2], args[3], SecondaryMass=10, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(args[1]),mpirank=rank, solcounter=solcount,nsols=nsols)
parallel_func = lambda args,solcount,nsols: process(args[0], args[1], args[2], args[3], SecondaryMass=10, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(args[1]), solcounter=solcount,nsols=nsols)

coords = [(smbhmass, smbhspin, pmass,e0)]
"""
def split(a, n):
        k, m = divmod(len(a), n)
        return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

split_parallel_args = split(coords, comm.Get_size())
parallel_args_for_subprocesses = comm.scatter(split_parallel_args,root=0)
counter=1;
if rank==0:
        print("Size of parameter space: {0}\nNumber MPI subprocesses: {1}".format(len(coords), comm.Get_size()), file=stdout_file)
        print("shape of partitioned parameter space: {0}".format(np.shape(split_parallel_args)), file=stdout_file)
with open("Rank{0}ProcessArguments.dat".format(rank), "w+") as file:
        for inx, val in enumerate(parallel_args_for_subprocesses):
                file.write("inx: {0}     val: {1}\n".format(inx+1,val))
for inx, arg in enumerate(parallel_args_for_subprocesses):
        parallel_func(arg,counter,len(parallel_args_for_subprocesses))
        counter+=1
"""


ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")



DataDir = DataDirectory

tmparr = np.linspace(1,9,9,dtype=np.int64) #strange floating point error when doing just np.arange(1,10,0.1) for np.linspace(1,10,91). Causes issues when saving numbers to filenames
tmparr1 = np.linspace(1,9,81, dtype=np.float64)
SMBHMasses = sorted([int(i) for i in np.kron(tmparr,[1e5, 1e6,1e7])]) #solar masses
SMBHSpins = [int(100*i)/100 for i in np.linspace(0.6,0.9,10)]
SecondaryMass = 10 #solar masses
e0list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
ProcaMasses = [round(i,22) for i in np.kron(tmparr1, [1e-16,1e-17,1e-18,1e-19])] #eV   #again avoiding floating point errors

maxlen = len(SMBHMasses)*len(SMBHSpins)*len(ProcaMasses)*len(e0list)
counter=1
for bh in SMBHMasses:
    for a in SMBHSpins:
        for pm in ProcaMasses:
            for e0 in e0list:
                alphaval = alphavalue(bh, pm)

                if alphaval>BHSpinAlphaCutoff(a) or alphaval<0.02:
                    continue

                p0 = GetInitialP(bh,e0)

                print("On iteration {0} out of {1}\n\t BHMass: {2}\n\tBHSpin: {3}\n\tPMass: {4}\n\te0: {5}\n\tp0: {6}\n\tT: {7}".format(counter, maxlen, bh, a, pm,e0,p0,T))
                unmoddedwvcl = EMRIWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)
                moddedwvcl = EMRIWithProcaWaveform(inspiral_kwargs=inspiral_kwargs.copy(), sum_kwargs=sum_kwargs.copy(), use_gpu=False)

                ProcaSolution.__init__(moddedwvcl,bh, a, pm, BosonSpin=1, CloudModel="relativistic", units="physical",UltralightBoson=ulb)
                Kerr.__init__(moddedwvcl,BHSpin=a)


                if e0<1e-6:
                    warnings.warn("Eccentricity below safe threshold for FEW. Functions behave poorly for e<1e-6")
                    e0=1e-6 #Certain functions in FEW are not well-behaved below this value

                OrbitalConstantsChange = moddedwvcl.ChangeInOrbitalConstants(SecondaryMass=10, SMBHMass=bh)
                asymptoticBosonCloudEFlux = OrbitalConstantsChange["E"] #Dimensionfull Flux. Mass Ratio prefactor comes from derivative of orbital energy wrt spacetime mass and factor of mass of the geodesic. Takes into account effective mass seen by secondary BH during its orbit
                asymptoticBosonCloudLFlux = OrbitalConstantsChange["L"]


                moddedwvcl.inspiralkwargs["DeltaEFlux"] = asymptoticBosonCloudEFlux
                moddedwvcl.inspiralkwargs["DeltaLFlux"] = asymptoticBosonCloudLFlux
                moddedwvcl.inspiralkwargs["FluxName"] = "analytic"

                unmoddedtraj = unmoddedwvcl.inspiral_generator(bh,10,a,p0,e0,1.,T=T, dt=dt, Phi_phi0=0, Phi_theta0=0, Phi_r0=0, **unmoddedwvcl.inspiralkwargs)

                moddedtraj = moddedwvcl.inspiral_generator(bh,10,a,p0,e0,1.,T=T, dt=dt, Phi_phi0=0, Phi_theta0=0, Phi_r0=0, **moddedwvcl.inspiralkwargs)
                counter+=1
