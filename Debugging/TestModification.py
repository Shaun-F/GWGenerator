import os, sys
os.chdir("../")
path = os.getcwd()
sys.path.insert(0, path)
import GWGen
from GWGen import *
import matplotlib.pyplot as plt

M=1e6
m=1e0
mu=2.8e-18
a = 0.9
p0=5
e0=0.5
x0=1

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False

T=5
dt=30
print("Alpha value: {0}".format(alphavalue(M,mu)))
print("Calculating unmodded waveform")
wvcl = EMRIWaveform()
unmodded = wvcl(M, m, a, p0, e0, x0, qS, phiS, qK, phiK, dist, Phi_phi0=0, Phi_theta0=0, Phi_r0=0, mich=mich, dt=dt, T=T)

print("calculating modded waveform")
mwvcl = EMRIWithProcaWaveform()
modded = mwvcl(M, m, mu, a, p0, e0, x0, T=T, qS=qS, phiS=phiS, qK=qK, phiK=phiK, dist=dist, mich=mich)




############### plot #################
fig,axes = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)
fig.set_size_inches(16,8)


t = np.arange(len(unmodded)) * dt
axes[0,0].set_title("Unmodded Model")
axes[0,0].plot(t, unmodded.real);
xticks = axes[0,0].get_xticks()[1:-1]
axes[0,0].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*365))]);
axes[0,0].set_xlabel("years");
axes[0,0].set_ylabel("strain");


tf =np.arange(len(modded)) * dt;
axes[0,1].set_title("modded Model")
axes[0,1].plot(tf, modded.real);
xticks = axes[0,1].get_xticks()[1:-1]
axes[0,1].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*365))]);
axes[0,1].set_xlabel("years");
axes[0,1].set_ylabel("strain");

axes[1,0].set_title("Overlap")
startinginx = 500000
cutoffinx = 501000;
axes[1,0].plot(t[startinginx:cutoffinx], unmodded.real[startinginx:cutoffinx]);
axes[1,0].plot(tf[startinginx:cutoffinx], modded.real[startinginx:cutoffinx]);
xticks = axes[1,0].get_xticks()[1:-1]
axes[1,0].set_xticks(xticks, [int(i)/100 for i in (xticks*100/(60*60*24*31))]);
axes[1,0].set_xlabel("months");
axes[1,0].set_ylabel("strain");

axes[1,1].text(0.5,0.5, "Mismatch = {0}".format(get_mismatch(unmodded, modded)))
axes[1,1].axis("off")

plt.show()
