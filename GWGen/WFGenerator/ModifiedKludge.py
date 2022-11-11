from .Kludge import *
from GWGen.numerical_flux import *
from GWGen.DressedFluxes import *



class FluxFunction(name="analytic"):
    def __init__(self):
        if name=="analytic":
            self.EFlux = Analytic5PNEFlux
            self.LFlux = Analytic5PNLFlux
        elif name=="numerical":
            self.EFlux = GenerateNumericalEInterpolation()
            self.LFlux = GenerateNumericalEInterpolation()

class ModifiedKludgeWaveform():
    def __init__(self):
        pass
