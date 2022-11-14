import GWGen
import numpy as np
import matplotlib.pyplot as plt



if __name__==__main__:
    #run analysis

    tmparr = np.arange(1,10,1)
    SMBHMasses = np.kron(tmparr,[1e4, 1e5, 1e6]) #solar masses
    SecondaryMass = 1 #solar masses
    ProcaMasses = np.kron(tmparr, [1e-14,1e-15,1e-16,1e-17]) #eV

    ALPHACUTOFF = 0.02 #cutoff for dimensionless gravitational coupling. values larger than this correspond to proca clouds whose GW fluxes exceed that of the EMRI

    for BHMASS in SMBHMasses:
        for PROCAMASS in ProcaMasses:
            #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
            if alphavalue(BHMASS,PROCAMASS)>0.02:
                continue
            
