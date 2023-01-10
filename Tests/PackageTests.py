import unittest
import GWGen
from GWGen.WFGenerator import *
from GWGen.DressedFluxes import *
import superrad
from joblib import Parallel, delayed
import numpy as np

class TestMethods(unittest.TestCase):

    def __init__(self):
        self.M_test = 1e6 #SMBHMass
        self.m_test = 1e1 #secondary bh mass
        self.mu_test = 1e-17 #proca mass (ev)
        self.murange_test = [0.81e-17, 4.45e-17] #proca mass range
        self.a_test = 0.9 #SMBH Spin
        self.p0_test = 10. #initial semi-lat rectum
        self.e0_test = 0.5 #initial eccentricity
        self.x0_test = 1. #inclination
        self.T_test = 5 #waveform time in years
        self.dist_test = 1. #distance in Mpc

        self.ulb = superrad.ultralight_boson.UltralightBoson(spin=1, model="relativistic")
        self.pc = ProcaSolution(self.M_test, self.a_test,self.mu_test, UltralightBoson = self.ulb)

    def test_LowEnergyLimit(self):
        """
            Verify low energy limit of proca cloud approachs bare black hole waveform
        """
        M = self.M_test
        m = self.m_test
        murange = self.murange_test[:10]
        mudelta = 0.1
        bhspin = self.a_test
        p0=self.p0_test
        e0=self.e0_test
        T=self.T_test
        qS = 1e-20
        phiS = 0.
        qK = 0.
        phiK = 0.
        dist=self.dist_test
        inspiral_kwargs = {"npoints":100, "max_init_len":1e3}
        sum_kwargs = {"use_gpu":False, "pad_output":False}

        wvcls = EMRIWaveform(inspiral_kwargs = inspiral_kwargs.copy(), sum_kwargs = sum_kwargs.copy(), use_gpu=False)
        procawvcls = EMRIWithProcaWaveform(inspiral_kwargs = inspiral_kwargs.copy(), sum_kwargs = sum_kwargs.copy(), use_gpu=False)
        bare_waveform = wvcls(M, m,bhspin,p0,e0,1., qS,phiS,qK,phiK,dist, T=T,npoints=inspiral_kwargs["npoints"])
        def process(procamass):
            print("Calculating proca mass {0}.".format(procamass))
            proca_waveform = procawvcls(M,m,procamass, bhspin, p0,e0, 1.,T=T, npoints = inspiral_kwargs["npoints"],UltralightBoson=self.ulb)
            mismatch = get_mismatch(bare_waveform, proca_waveform)
            return mismatch

        results = [process(i) for i in np.arange(murange[0], murange[-1], 0.1e-17)]
        boolresult = (sorted(results)==results)

        self.LowEnergyLimitResults = results
        self.assertTrue(boolresult)

    def test_ProcaFluxSign(self):
        """
            Verify sign of energy and angular momentum flux is correct
        """
        mudom = np.arange(self.murange_test[0], self.murange_test[-1], 0.05e-17)
        procaenergyflux = [ProcaSolution(self.M_test, self.a_test, i, UltralightBoson = self.ulb).BosonCloudGWEFlux(0) for i in mudom]
        procaangmomflux = [ProcaSolution(self.M_test, self.a_test, i, UltralightBoson = self.ulb).BosonCloudGWLFlux(0) for i in mudom]
        res = np.logical_and(np.all([i<0 for i in procaenergyflux]), np.all([i<0 for i in procaangmomflux]))

        self.assertTrue(res)
