import time
import numpy as np
import scipy as sp
from scipy import interpolate
import scipy.fft
import warnings
from mpmath import *
import re #must be imported after mpmath to override definitions in mpmath package
import astropy.constants as cons
import astropy.units as unit
import glob
mp.dps=25
mp.pretty=True

#predefined regex expressions
alpha_match_1 = re.compile("Alpha_\d+_\d+")
alpha_match_2 = re.compile("Alpha_\d+")
modeovertonematch = re.compile("Mode_1_Overtone_0")
alpha_rege = re.compile("\d+")


def GetInitialP(SMBHMass, InitialEccentricity):
    """
    estimate initial semi-latus rectum for given BH mass and initial eccentricity such that the coalescence occurs after 5 years
    """
    y = [1e5,3e5, 4e5, 5e5, 1e6,3e6,5e6,1e7]
    x = [0.1,0.3,0.5,0.7,0.8]
    z = [[35.1,  20.1,17.4,15.6, 11.1,    7.,   5.7, 4.83],
        [34.8, 20, 17.3, 15.5, 11, 6.97, 5.55, 4.83],
        [34.2, 19.6, 17, 15.2, 10.9, 6.9, 5.8, 4.76],
        [32.5, 18.7, 16.25, 14.56, 10.47, 6.64, 5.53, 4.46],
        [30.9, 17.9, 15.5, 13.9, 10.0, 6.27, 4.9, 4.3]]
    interp = sp.interpolate.RectBivariateSpline(x,y,z, kx=1, ky=1,s=0)

    retvalue = interp(InitialEccentricity, SMBHMass)
    return retvalue[0][0]


#Increase density of points for input array by specifying total number of output points
def IncreaseArrayDensity(arr, npoints):
    assert len(arr)<=npoints, "Error: npoints must be equal to or larger than length of input array"
    if npoints>len(arr):
        newarr = [arr[0]]
        remainder = npoints*len(arr)
        for inx, i in enumerate(arr):
            if inx<len(arr)-1:
                newintpnts = int(floor(npoints/len(arr)))+1
                newsec = np.linspace(arr[inx], arr[inx+1], newintpnts)[1:]
                _=[newarr.append(i) for i in newsec]
        remainderpoints=npoints-len(newarr)
        arrend = newarr[-remainderpoints-2:]
        arrendavgs = 1/2*(np.array(arrend)[1:] + np.array(arrend)[:-1])
        newarrend = []
        for i in range(len(arrend) + len(arrendavgs)):
            if i%2 == 0:
                newarrend.append(arrend[int(i/2)])
            if i%2 == 1:
                newarrend.append(arrendavgs[int((i-1)/2)])
        newarr = newarr[:-remainderpoints-3]
        _=[newarr.append(i) for i in newarrend]
        return newarr
    elif npoints==len(arr):
        return arr



#extract alpha value from filename
def AlphaValFromFilename(filename):
    alphastr = alpha_match_1.findall(filename)
    if len(alphastr)==0:
        alphastr = alpha_match_2.findall(filename)
    if isinstance(alphastr,str):
        pass
    if isinstance(alphastr,list):
        alphastr=alphastr[0]
    alphapieces = alpha_rege.findall(alphastr)
    if len(alphapieces)>1:
        alphavalue = float(alphapieces[0])/float(alphapieces[1])
    else:
        alphavalue = float(alphapieces[0])
    return alphavalue


#efficient cartesian product of two arrays
## https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def ConvertToCCompatibleArray(arr,newdtype=None):
    if not newdtype==None:
        if isinstance(arr,np.ndarray):
            ret = np.require(arr, dtype=arr.dtype, requirements=["C", "O", "A", "E"])
        elif isinstance(arr,list):
            ret = np.require(arr, dtype=type(arr[0]), requirements=["C", "O", "A", "E"])
        else:
            raise TypeError("Input array must be either a list or a numpy array object")
    else:
        ret = np.require(arr, dtype=newdtype, requirements=["C", "O", "A", "E"])
    return ret

def WaveformInnerProduct(timedomain, h1,h2, fmin=0.0001, fmax=1):
    """
    complex waveforms h1 and h2 are in time-domain with time domain in units of seconds. Compute inner product defined in
    """
    if len(h1)!=len(h2):
        warnings.warn("Waveforms have different lengths. Truncating longer waveform")
        minlength = min([len(h1), len(h2)])-1
        h1 = h1[0:minlength]
        h2 = h2[0:minlength]

    if len(timedomain)!=len(h1) or len(timedomain)!=len(h2):
        raise RuntimeError("time domain has different length than the waveforms")

    h1_InFreq = sp.fft.fft(h1)
    h2_InFreq = sp.fft.fft(h2)
    timelength = len(timedomain)
    DeltaT = timedomain[1]-timedomain[0]
    frequency_range = sp.fft.fftfreq(timelength, d=DeltaT)

    #Consider real frequencies and ignore zero frequency
    frequency_length = int(len(frequency_range)/2 -1)
    frequency_domain = frequency_range[1:frequency_length]
    h1f = h1_InFreq[1:frequency_length]
    h2f = h2_InFreq[1:frequency_length]
    h2fstar = np.conjugate(h2f)
    PowerSpectralDensity = LisaSensitivity(frequency_domain)



    integrand = h1f*h2fstar/PowerSpectralDensity


    if frequency_domain[0]<fmin:
        greater_mask = np.ma.masked_greater(frequency_domain,fmin).mask
    else:
        greater_mask = np.array([True for i in frequency_domain])

    if frequency_domain[-1]>fmax:
        lesser_mask = np.ma.masked_less(frequency_domain,fmax).mask
    else:
        lesser_mask = np.array([True for i in frequency_domain])

    mask = np.logical_not(np.logical_and(greater_mask,lesser_mask))
    masked_frequency_range = np.ma.masked_where(mask,frequency_domain).compressed()
    masked_integrand = np.ma.masked_where(mask, integrand).compressed()



    integral = sp.integrate.simpson(masked_integrand.real, x=masked_frequency_range)

    ret = 4*integral
    return ret

def Faithfulness(timedomain, h1, h2):
    """
    time domain must be in units of seconds
    """

    if len(h1)!=len(h2):
        warnings.warn("Waveforms have different lengths. Truncating longer waveform")
        minlength = min([len(h1), len(h2)])
        h1 = h1[0:minlength]
        h2 = h2[0:minlength]
        assert len(h1)==len(h2)

    assert len(timedomain)==len(h1), "time domain has different length than the waveforms. time domain length: {0} waveform 1 length: {1} waveform 2 length: {2}".format(len(timedomain), len(h1), len(h2))


    h1h2 = WaveformInnerProduct(timedomain, h1, h2)
    h1h1 = WaveformInnerProduct(timedomain, h1, h1)
    h2h2 = WaveformInnerProduct(timedomain, h2, h2)
    ret = h1h2/np.sqrt(h1h1*h2h2)

    return ret


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
    return ret

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





def PrettyPrint(str):
    decoration = "****************"
    prn_str = decoration+'\n'+str+'\n'+decoration
    print(prn_str)
