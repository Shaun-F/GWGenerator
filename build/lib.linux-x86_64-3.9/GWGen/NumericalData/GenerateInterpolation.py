import scipy as sp
import pandas as pd
import numpy as np
import os


colnames=["q","p","e","theta","E","Lz","C","Einf","LzInf","Cinf","Eh","Lzh","Ch","pinf","einf","thetainf","ph","eh","thetah","lmax","DeltaEinf","DeltaEh"]

#we consider only prograde orbits
def GenerateNumericalEInterpolation():
    path = os.path.dirname(__file__);
    DataSpinGetter = lambda spin: pd.read_csv(path+"/dIdt_q"+spin+"inc0.dat", header=None, delimiter=' ', names=colnames)
    alldat = [DataSpinGetter(spin) for spin in ['0.10', '0.30', '0.50', '0.70', '0.90']]
    UnifiedDataFrame = pd.concat(alldat)
    coords = UnifiedDataFrame[{"q", "e", "p"}].values
    dat = UnifiedDataFrame["Einf"].values
    interpFUN = sp.interpolate.RBFInterpolator(coords, dat, neighbors=100, kernel="linear",smoothing=0, epsilon=6)
    return lambda q,e,p: interpFUN([[e,q,p]])

def GenerateNumericalLInterpolation():
    path = os.path.dirname(__file__);
    DataSpinGetter = lambda spin: pd.read_csv(path+"/dIdt_q"+spin+"inc0.dat", header=None, delimiter=' ', names=colnames)
    alldat = [DataSpinGetter(spin) for spin in ['0.10', '0.30', '0.50', '0.70', '0.90']]
    UnifiedDataFrame = pd.concat(alldat)
    coords = UnifiedDataFrame[{"q", "e", "p"}].values
    dat = UnifiedDataFrame["LzInf"].values
    interpFUN = sp.interpolate.RBFInterpolator(coords, dat, neighbors=100, kernel="linear",smoothing=0, epsilon=6)
    return lambda q,e,p: interpFUN([[e,q,p]])
