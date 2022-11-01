import pandas as pd
import numpy as np
import os
numericaldatapath = os.path.abspath(os.path.dirname(__file__));

def import_numerical_data(spin=0.9):
    available_spins=np.asarray([-0.9,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,0.9])
    colnames=["q", "p","e","theta","E","Lz","C","Einf","LzInf", "Cinf","Eh","Lzh","Ch","pinf","einf","thetainf","ph","eh","thetah","lmax","DeltaEinf","DeltaEh"]
    nearest_spin=available_spins[(np.abs(spin-available_spins).argmin())];
    import_string = numericaldatapath+'/../../numerical_flux/dIdt_q'+str(nearest_spin)+'0inc0.dat';
    return pd.read_csv(import_string,delimiter=' ', header=None, names=colnames)
