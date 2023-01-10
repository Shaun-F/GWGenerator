import unittest
import GWGen
from GWGen.WFGenerator import *
import superrad
from joblib import Parallel, delayed
import numpy as np

class TestMethods(unittest.TestCase):

    def test_LowEnergyLimit(self):
        M = 1e6
        m = 10
        murange = [0.81e-17, 1.81e-17]
        mudelta = 0.1
        bhspin = 0.9
        p0=10
        e0=0.5
        T=5
        qS = 1e-20
        phiS = 0.
        qK = 0.
        phiK = 0.
        dist=1.
        ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")
        inspiral_kwargs = {"npoints":100, "max_init_len":1e3}
        sum_kwargs = {"use_gpu":False, "pad_output":False}

        wvcls = EMRIWaveform(inspiral_kwargs = inspiral_kwargs.copy(), sum_kwargs = sum_kwargs.copy(), use_gpu=False)
        procawvcls = EMRIWithProcaWaveform(inspiral_kwargs = inspiral_kwargs.copy(), sum_kwargs = sum_kwargs.copy(), use_gpu=False)
        bare_waveform = wvcls(M, m,bhspin,p0,e0,1., qS,phiS,qK,phiK,dist, T=T,npoints=inspiral_kwargs["npoints"])
        def process(procamass):
            print("Calculating proca mass {0}.".format(procamass))
            proca_waveform = procawvcls(M,m,procamass, bhspin, p0,e0, 1.,T=T, npoints = inspiral_kwargs["npoints"],UltralightBoson=ulb)
            mismatch = get_mismatch(bare_waveform, proca_waveform)
            return mismatch

        results = [process(i) for i in np.arange(murange[0], murange[-1], 0.1e-17)]
        boolresult = (sorted(results)==results)

        self.LowEnergyLimitResults = results
        self.assertTrue(boolresult)
