import GWGen
import numpy as np
import matplotlib.pyplot as plt



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

if __name__==__main__:
    #run analysis

    tmparr = np.arange(1,10,1)
    SMBHMasses = np.kron(tmparr,[1e6]) #solar masses
    SecondaryMass = 10 #solar masses
    ProcaMasses = np.kron(tmparr, [1e-14,1e-15,1e-16,1e-17,1e-18,1e-19]) #eV

    ALPHACUTOFF = 0.02 #cutoff for dimensionless gravitational coupling. values larger than this correspond to proca clouds whose GW fluxes approximately exceed that of the EMRI

    for inx, BHMASS in enumerate(SMBHMasses):
        for inx2,PROCAMASS in enumerate(ProcaMasses):
            #alpha values larger than 0.02 produce energy fluxes larger than the undressed flux
            if alphavalue(BHMASS,PROCAMASS)>0.02:
                continue
            unmoddedtraj
            unmoddedwvcl = EMRIWaveform()
            unmoddedwv = unmoddedwvcl(BHMASS, SecondaryMass, BHSpin, p0, e0, x0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt,T=T)

            moddedwvcl = EMRIWithProcaWaveform()
            moddedwv = moddedwvcl(BHMASS, SecondaryMass, PROCAMASS, BHSpin,p0,e0,x0,T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist,mich=mich)
