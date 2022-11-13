from .Kludge import *
from .. import NumericalData, DressedFluxes
from ..NumericalData import *
from ..DressedFluxes import *
from ..UndressedFluxes import FluxFunction
from few.waveform import AAKWaveformBase


class ModifiedKludgeWaveform(ProcaSolution,AAKWaveformBase, Kerr):
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

    def __call__(self, SMBHMass, SecondaryMass, ProcaMass, BHSpin, p0, e0, x0, T=1, npoints=10, BosonSpin=1, CloudModel="relativistic", units="physical", FluxName="analytic"):
        massRatio = SecondaryMass/SMBHMass
        ProcaSolution.__init__(self,SMBHMass, BHSpin, ProcaMass, BosonSpin=BosonSpin, CloudModel=CloudModel, units=units)
        Kerr.__init__(self,BHSpin=BHSpin)


        if e0>1e-16:
            raise NotImplementedError("Non-circular orbits not implemented for modified kludge")
        if e0==0:
            e0=1e-16
        asymptoticBosonCloudEFlux = 1/massRatio * self.GWFlux()
        asymptoticBosonCloudLFlux = asymptoticBosonCloudEFlux/self.OmegaPhi(e, p)

        self.inspiralkwargs["DeltaEFlux"] = asymptoticBosonCloudEFlux
        self.inspiralkwargs["DeltaLFlux"] = asymptoticBosonCloudLFlux
        self.inspiralkwargs["FluxName"] = FluxName
