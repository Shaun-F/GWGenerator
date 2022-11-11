import scipy as sp
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()

colnames=["q","p","e","theta","E","Lz","C","Einf","LzInf","Cinf","Eh","Lzh","Ch","pinf","einf","thetainf","ph","eh","thetah","lmax","DeltaEinf","DeltaEh"]

#we consider only prograde orbits
def GenerateNumericalEInterpolation():
    DataSpinGetter = lambda spin: pd.read_csv(os.getcwd()+"/dIdt_q"+spin+"inc0.dat", header=None, delimiter=' ', names=colnames).values
    alldat = [DataSpinGetter(spin) for spin in ['0.10', '0.30', '0.50', '0.70', '0.90']]
    UnifiedDataFrame = pd.concat(alldat)
    coords = unified[{"q", "e", "p"}].values
    dat = unified["Einf"].values
    interpFUN = sp.interpolate.RBFInterpolator(coords, dat, neighbors=100, kernel="linear",smoothing=0, epsilon=6)
    return lambda q,e,p: interpFUN([[q,e,p]])

def GenerateNumericalLInterpolation():
    DataSpinGetter = lambda spin: pd.read_csv(os.getcwd()+"/dIdt_q"+spin+"inc0.dat", header=None, delimiter=' ', names=colnames).values
    alldat = [DataSpinGetter(spin) for spin in ['0.10', '0.30', '0.50', '0.70', '0.90']]
    UnifiedDataFrame = pd.concat(alldat)
    coords = unified[{"q", "e", "p"}].values
    dat = unified["Linf"].values
    interpFUN = sp.interpolate.RBFInterpolator(coords, dat, neighbors=100, kernel="linear",smoothing=0, epsilon=6)
    return lambda q,e,p: interpFUN([[q,e,p]])
