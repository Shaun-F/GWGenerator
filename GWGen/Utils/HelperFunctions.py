import re
from mpmath import *
import astropy.constants as cons
import astropy.units as unit
mp.dps=25
mp.pretty=True


def WaveformInnerProduct(timedomain, h1,h2, fmin=0.0001, fmax=1):
    """
    complex waveforms h1 and h2 are in time-domain with time domain in units of seconds. Compute inner product defined in
    """
    h1_InFreq = np.fft.fft(h1)
    h2_InFreq = np.fft.fft(h2)
    timelength = len(timedomain)
    DeltaT = timedomain[1]-timedomain[0]
    frequency_range = np.fft.fftfreq(timelength, d=DeltaT)

def LisaSensitivity(f):
    """
    frequency in Hertz

    http://arxiv.org/abs/1803.01944
    """
    L=2.5e9 #gigameters
    fstar = 0.01909 #Hertz
    ret = (10/(3*L**2)) * (OpticalMetrologyNoise(f) + 4*AccelNoise(f)/((2*np.pi*f)**4))*(1+(6/10)*(f/fstar)**2) + ConfusionNoise(f)
    return ret


def OpticalMetrologyNoise(f):
    """
    frequency in Herz

    http://arxiv.org/abs/1803.01944
    """
    ret = (1.5*10**(-11))**2 * (1 + (0.002/f)**4) # square meters per Hertz
    return ret

def AccelNoise(f):
    """
    frequency in Hertz

    http://arxiv.org/abs/1803.01944
    """
    ret = (3*10**(-15))**2 * (1+(0.0004/f)**2)*(1+(f/0.008)**4) #square meters per quartic seconds per Hertz
    return ret

def ConfusionNoise(f):
    """
    frequency in Hertz

    http://arxiv.org/abs/1803.01944
    """
    #4 year observation time
    A = 9*10**(-45)
    alpha = 0.138
    beta = -221
    gamma = 1680
    kappa = 521
    fk = 0.00113
    ret = A*f**(-7/3) * np.exp(-f**alpha + beta*f*np.sin(kappa*f)) * (1+np.tanh(gamma*(fk-f))) #inverse Hertz

def stringtocomplex(string):
    exponentpiece = re.search(r'\^-\d+', string).group()
    exponent = float(re.search('-\d+', exponentpiece).group())
    complexpartNumber = re.search(r'\d+\.\d+\*\^\-\d+\*j', string).group()[0:-6]
    complexpartSign = re.search(r' \W ', string).group()[1];
    complexpart = float(complexpartSign+complexpartNumber)*10**(exponent)
    realpart = float(re.search(r'\S+\d+.\d+', string).group())

    return complex(realpart, complexpart)


###mathematica namespace converter

def Sqrt(x):
    return x**(1/2)

def Power(x,y):
    return x**y

def Abs(x):
    return abs(x)


#complete elliptic integral of the 1st kind
def EllipticK(m):
    try:
        return(float(ellipk(m)))
    except TypeError:
        print("ERROR: input {0} returns imaginary result".format(m))
#complete elliptic integral of the 2nd kind
def EllipticE(m):
    try:
        return(float(ellipe(m)))
    except TypeError:
        print("ERROR: input {0} returns imaginary result".format(m))

#complete elliptic integral of the 3rd kind
def EllipticPi(n,m):
    try:
        return(float(ellippi(n,m)))
    except TypeError:
        print("ERROR: input {0},{1} returns imaginary result".format(n,m))




def alphavalue(BHMass, procamass):
    if procamass>1e-10:
        print("Is proca mass in units of eV?")
    if BHMass<1:
        print("Is Black hole mass in units of solar masses?")
    return (procamass*unit.eV*BHMass*unit.Msun*cons.G/(cons.hbar*cons.c**3)).decompose()
