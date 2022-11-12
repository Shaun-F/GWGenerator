from .Kludge import *
from .. import NumericalData, DressedFluxes
from ..NumericalData import *
from ..DressedFluxes import *



class FluxFunction():
    def __init__(self,name="analytic"):
        if name=="analytic":
            self.EFlux = Analytic5PNEFlux
            self.LFlux = Analytic5PNLFlux
        elif name=="numerical":
            self.EFlux = GenerateNumericalEInterpolation()
            self.LFlux = GenerateNumericalLInterpolation()

    def __call__(self,q,e,p):
        return {"EFlux":self.EFlux(q,e,p),"LFlux":self.LFlux(q,e,p)}

class ModifiedKludgeWaveform():
    def __init__(self):
        pass
