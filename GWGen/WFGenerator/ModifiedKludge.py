from .Kludge import *
from .. import NumericalData, DressedFluxes
from ..NumericalData import *
from ..DressedFluxes import *
from ..UndressedFluxes import FluxFunction
from few.waveform import AAKWaveformBase


class EMRIWithProcaWaveform(ProcaSolution,AAKWaveformBase, Kerr):
    def __init__(self,
                    inspiralfunction_kwargs={},
                    summationfunction_kwargs={},
                    use_gpu=False,
                    num_threads=None
                ):

        self.inspiralkwargs = inspiralfunction_kwargs
        self.sumkwargs = summationfunction_kwargs
        self.use_gpu = use_gpu
        self.num_threads = num_threads

    def __call__(self, SMBHMass, SecondaryMass, ProcaMass, BHSpin, p0, e0, x0, T=1, npoints=10, BosonSpin=1, CloudModel="relativistic", units="physical", FluxName="analytic", **kwargs):
        massRatio = SecondaryMass/SMBHMass
        qs = kwargs.get("qS", 0)
        phis = kwargs.get("phiS", 0)
        qk = kwargs.get("qK", 0)
        phik = kwargs.get("phiK", 0)
        dist = kwargs.get("dist", 1)
        Phi_phi0 = kwargs.get("Phi_phi0", 0)
        Phi_theta0 = kwargs.get("Phi_theta0",0)
        Phi_r0 = kwargs.get("Phi_r0", 0)
        mich = kwargs.get("mich", False)
        dt = kwargs.get("dt", 15)
        T = kwargs.get("T",1)

        ProcaSolution.__init__(self,SMBHMass, BHSpin, ProcaMass, BosonSpin=BosonSpin, CloudModel=CloudModel, units=units) #How to use super() with multiple inheritance with different positional arguments for each __init__?
        Kerr.__init__(self,BHSpin=BHSpin)


        if e0>1e-6:
            raise NotImplementedError("Non-circular orbits not implemented for modified kludge")
        if e0==0:
            e0=1e-6 #Certain functions in FEW are not well-behaved below this value
        asymptoticBosonCloudEFlux = self.BosonCloudGWFlux()
        asymptoticBosonCloudLFlux = "ToFILL !!!!!!!!!!" 

        self.inspiralkwargs["DeltaEFlux"] = asymptoticBosonCloudEFlux
        self.inspiralkwargs["DeltaLFlux"] = asymptoticBosonCloudLFlux
        self.inspiralkwargs["FluxName"] = FluxName

        aakwaveform = AAKWaveformBase(PNTraj,
                                        AAKSummation,
                                        inspiral_kwargs=self.inspiralkwargs,
                                        sum_kwargs = self.sumkwargs,
                                        use_gpu=self.use_gpu,
                                        num_threads=self.num_threads)

        return aakwaveform(SMBHMass, SecondaryMass, BHSpin, p0, e0, x0, qs,phis,qk,phik, dist, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
