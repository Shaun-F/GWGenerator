import os, sys
os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen.Utils import *
from GWGen.WFGenerator import *
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
import superrad


# set initial parameters
M = 1e5
m = 1e1
mu = 2.8e-16
e0 = 0.5
p0 = GetInitialP(M,e0)
Phi_phi0 = 0.
Phi_theta0 =0.
Phi_r0 = 0.


a=0.9 #SMBH Spin
Y0=1. #Initial Inclincation
qS=np.pi/4 #Sky Location Polar Angle in solar system barycenter coordinate system
phiS=0. #Sky Location Azimuthal Angle in solar system barycenter coordinate system
qK=1e-6 #Initial BH Spin Polar Angle in solar system barycenter coordinate system
phiK=0. #Initial BH Spin Azimuthal Angle in solar system barycenter coordinate system
dist=1. #Distance to source (Mpc)
mich=False #assume LISA long baseline response approximation

T=5.9 #LISA data run is 5 years. We set the max time to be longer because the proca cloud extends the inspiral time
dt=15 #time resolution in seconds

alphaval = alphavalue(M,mu)



use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
insp_kwargs = {
    "npoints": 110,  # we want a densely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "dense_output":True
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

def innerprod4(td,w1,w2):
    wv1fft = sp.fft.fft(w1)
    wv2fft = sp.fft.fft(w2)
    freqs = sp.fft.fftfreq(len(td), d=float(td[1]-td[0]))
    wv1fft = wv1fft[1:]
    wv2fft = wv2fft[1:]
    freqs = freqs[1:]
    lisasens = LisaSensitivity(np.abs(freqs))
    t1 = sp.fft.ifft(wv1fft/lisasens)
    t2 = sp.fft.ifft(np.conjugate(wv2fft))
    conv = sp.signal.convolve(t1,t2,method="fft", mode="full")
    convlen = int(len(conv))+1
    #plt.plot(conv)
    #plt.scatter([convlen/2], [conv[int(convlen/2)]])
    if np.all(w1==w2):
        return 2*np.real(conv[int(convlen/2)])
    return 2*np.real(conv)

WithoutProcaInspiralKwargs = insp_kwargs.copy()
WithoutProcaSumKwargs=sum_kwargs.copy()
withoutprocagen = EMRIWaveform(inspiral_kwargs=WithoutProcaInspiralKwargs, sum_kwargs=WithoutProcaSumKwargs, use_gpu=False)
ulb = superrad.ultralight_boson.UltralightBoson(spin=1,model="relativistic")
withprocagen = EMRIWithProcaWaveform(inspiral_kwargs=insp_kwargs.copy(),sum_kwargs=sum_kwargs.copy())
print(r"alpha = {0}".format(alphaval))
print("initial p = {0}".format(p0))



fai = []
for i in np.linspace(1.e-16,4.5e-16,30):
    print(alphavalue(M,i))
    print("wv1")
    wv1 = withoutprocagen(M, m, a, p0, e0, Y0, qS, phiS, qK, phiK, dist,Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
    print('wv2')
    wv2 = withprocagen(M,m,i,a,p0,e0,Y0,T=T,qS=qS,phiS=phiS,qK=qK,phiK=phiK,dist=dist,mich=mich, UltralightBoson=ulb)

    if len(wv1)<len(wv2):
        wv1 = np.pad(wv1, (0,len(wv2)-len(wv1)))
    elif len(wv2)<len(wv1):
        wv2= np.pad(wv2, (0,len(wv1)-len(wv2)))

    minlen = min([len(wv1), len(wv2)])
    td = np.arange(minlen)*dt
    #wv1 = wv1[:minlen]
    #wv2 = wv2[:minlen]
    print("Faithfulness")
    fai.append([alphavalue(M,i),
        Faithfulness(td,wv1,wv2)]
    )
    print("Proca mass: "+str(i)+"\n\tfaithfulness: "+str(fai[-1]))
fai = np.asarray(fai)

print(fai)

plt.plot(np.array(fai)[:,0],np.array(fai)[:,1])
plt.show()
