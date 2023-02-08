from main import *
from mpi4py import MPI


smbhmass = 200000
smbhspin = 0.63

"""
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
"""

DataDir = os.path.abspath(os.path.dirname(__file__)) + "/Data/"

#parallel_func = lambda args,solcount,nsols: process(args[0], args[1], args[2], args[3], SecondaryMass=10, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(args[1]),mpirank=rank, solcounter=solcount,nsols=nsols)
parallel_func = lambda args,solcount,nsols: process(args[0], args[1], args[2], args[3], SecondaryMass=10, DataDir=DataDir, alphauppercutoff=BHSpinAlphaCutoff(args[1]), solcounter=solcount,nsols=nsols)

coords = [(smbhmass, smbhspin, 6e-17,0.8),(smbhmass, smbhspin, 6e-17,0.9),(smbhmass, smbhspin, 7e-17,0.1),(smbhmass, smbhspin, 7e-17,0.2)]

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

parallel_func((200000.0,0.63,8e-17,0.5),1,1)
