import time
import os
import fractions
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import scipy.fft
from scipy.integrate._ivp.ivp import *
import warnings
from mpmath import *
import re #must be imported after mpmath to override definitions in mpmath package
from bisect import bisect_right
import astropy.constants as cons
import astropy.units as unit
import glob

#FFT specific packages
import pyfftw
import multiprocessing


try:
    import cupy as cp
    usingcupy=True
except (ImportError, ModuleNotFoundError) as e:
    usingcupy=False

mp.dps=25
mp.pretty=True

#predefined regex expressions
alpha_match_1 = re.compile("Alpha_\d+_\d+")
alpha_match_2 = re.compile("Alpha_\d+")
modeovertonematch = re.compile("Mode_1_Overtone_0")
number_rege = re.compile("\d+")
bhspin_rege = re.compile("BHSpin_\d+_\d+")


#LISA response functions (Arxiv: http://arxiv.org/abs/gr-qc/0310125   equation 15)

F1plus = lambda theta, phi, psi: (1/2)*(1 + np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) - np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
F1cross = lambda theta, phi, psi: (1/2)*(1 + np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) + np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)

F2plus = lambda theta, phi, psi: (1/2)*(1 + np.cos(theta)**2)*np.sin(2*phi)*np.cos(2*psi) + np.cos(theta)*np.cos(2*phi)*np.sin(2*psi)
F2cross = lambda theta, phi, psi: (1/2)*(1 + np.cos(theta)**2)*np.sin(2*phi)*np.sin(2*psi) - np.cos(theta)*np.cos(2*phi)*np.cos(2*psi)

def detector_response(complex_strain,viewingangles = [0,0,0]):
    wvplus, wvcross = complex_strain.real, complex_strain.imag
    theta, phi, psi = viewingangles

    F1plus_response = F1plus(theta, phi, psi)
    F1cross_response = F1cross(theta, phi, psi)
    F2plus_response = F2plus(theta, phi, psi)
    F2corss_response = F2cross(theta, phi, psi)

    response_1 = np.sqrt(3)/2 * (F1plus_response*wvplus + F1cross_response*wvcross)
    response_2 = np.sqrt(3)/2 * (F2plus_response*wvplus + F2corss_response*wvcross)

    return {"h1":response_1, "h2":response_2}





def ProcaDataNameGenerator(bhspin, alpha, mode, overtone):
    bhspin_fraction = fractions.Fraction.from_float(bhspin).limit_denominator(1000)
    alpha_fraction = fractions.Fraction.from_float(alpha).limit_denominator(1000)
    string = "BHSpin_"+str(bhspin_fraction.numerator)+"_"+str(bhspin_fraction.denominator)+"_Alpha_"+str(alpha_fraction.numerator)+"_"+str(alpha_fraction.denominator)+"_Mode_"+str(mode)+"_Overtone_"+str(overtone)+".npz"
    return string

def BHSpinAlphaCutoff(bhspin):
    bhspins_lastalpha = [(0.6, 0.165), (0.62, 0.175), (0.64, 0.18), (0.66, 0.19), (0.68, 0.2), (0.7, 0.205), (0.72, 0.215), (0.74, 0.225), (0.76, 0.235), (0.78, 0.245), (0.8, 0.26), (0.82, 0.27), (0.84, 0.285), (0.86, 0.3), (0.88, 0.315), (0.9, 0.335)]
    bhspins = np.array(bhspins_lastalpha)[:,0]
    inx = bisect_right(bhspins, bhspin)
    if inx==len(bhspins):
        inx-=1
    alpha_cutoff = bhspins_lastalpha[inx-1][1]
    return alpha_cutoff

def KerrFrequencyAbsoluteBoundary(ecc):
    """
    Gives the absolute semi-latus rectum lower cutoff to ensure Kerr Frequencies doesnt throw errors
    """
    EccentricitySemilatusrectumCutoff = [[0.9,3.34],
                 [0.8,3.31],
                 [0.7,3.08],
                 [0.6,2.96],
                 [0.5,2.84],
                 [0.4,2.72],
                 [0.3,2.61],
                 [0.2,2.51],
                 [0.1,2.41],
                 [1e-10,2.33]]
    return None #not currently implemented.

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
    alphapieces = number_rege.findall(alphastr)
    if len(alphapieces)>1:
        alphavalue = float(alphapieces[0])/float(alphapieces[1])
    else:
        alphavalue = float(alphapieces[0])
    return alphavalue

def BHSpinValFromFilename(filename):
    BHSpinSection = bhspin_rege.findall(filename)[0]
    numerator_denominator = number_rege.findall(BHSpinSection)
    assert len(numerator_denominator)>0, "Error: bhspin not found in file"

    value = float(numerator_denominator[0])/float(numerator_denominator[1])

    return value




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

def WaveformInnerProduct(timedomain, h1,h2, use_gpu=False, maximize=False, viewingangle = [0,0,0]):
    """
    complex waveforms h1 and h2 are in time-domain with time domain in units of seconds. Compute inner product defined in
    """

    assert h1.dtype=="complex" and h2.dtype=="complex", "Error: Input waveforms must be complex strain amplitudes, h = h_plus + i*h_cross"

    if use_gpu:
        xp = cp
        if isinstance(h1,np.ndarray):
            h1 = xp.asarray(h1)
        if isinstance(h2, np.ndarray):
            h2 = xp.asarray(h2)
    else:
        xp = np

        try:
            if isinstance(h1,cp.ndarray):
                h1 = xp.asarray(h1.get())
        except NameError:
            pass
        try:
            if isinstance(h2,cp.ndarray):
                h2 = xp.asarray(h2.get())
        except NameError:
            pass

    if len(h1)!=len(h2):
        warnings.warn("Waveforms have different lengths. Zero padding shorter waveform")
        if len(h1)<len(h2):
            h1 = np.pad(h1, (0,len(h2)-len(h1)))
        elif len(h2)<len(h1):
            h2 = np.pad(h2, (0, len(h1)-len(h2)))


    if len(timedomain)!=len(h1) or len(timedomain)!=len(h2):
        raise RuntimeError("time domain has different length than the waveforms")

    #maximize over coalescence http://arxiv.org/abs/1603.02444
    #sets phase of last element to zero, in accord with above paper
    if maximize:

        N_angles = 5
        sub_maxes = []
        for i in range(N_angles):
            h2 = h2*np.exp(1j* (i/N_angles) * 2 * np.pi) # Binary MUST be viewed face-on for this phase shift to work
            h1_responses = list(detector_response(h1, viewingangles = viewingangle).values())
            h2_responses = list(detector_response(h2, viewingangles = viewingangle).values())

            pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
            pyfftw.interfaces.cache.enable()
            with sp.fft.set_backend(pyfftw.interfaces.scipy_fft):
                with sp.fft.set_workers(os.cpu_count() + 2):
                    if not use_gpu:

                        h1resp_f = sp.fft.fft(h1_responses)[:,1:] #drop zero frequency
                        h2resp_f = sp.fft.fft(h2_responses)[:,1:]
                        h2resp_f_star = np.conjugate(h2resp_f)
                        timelength = len(timedomain)
                        DeltaT = timedomain[1]-timedomain[0]
                        frequency_range = sp.fft.fftfreq(int(timelength), d=float(DeltaT))[1:]
                    else:
                        h1resp_f = xp.fft.fft(h1_responses)[:,1:] #drop zero frequency
                        h2resp_f = xp.fft.fft(h2_resonses)[:,1:]
                        h2resp_f_star = np.conjugate(h2resp_f)
                        timelength = len(timedomain)
                        DeltaT = timedomain[1]-timedomain[0]
                        frequency_range = xp.fft.fftfreq(int(timelength), d=float(DeltaT))[1:]

                    PowerSpectralDensity = LisaSensitivity(np.abs(frequency_range))



                    Factor1 = xp.fft.ifft(h1resp_f/PowerSpectralDensity)
                    Factor2 = xp.fft.ifft(h2resp_f_star)
                    if use_gpu:
                        Factor1 = Factor1.get()
                        Factor2 = Factor2.get()

                    convolutions = sp.signal.convolve(Factor1, Factor2, method="fft", mode="full")[[0,-1]].real
                    combined_convolutions = np.sum(convolutions, axis=0)
                    submax = max(combined_convolutions)
                    sub_maxes.append(submax)

        absolute_max = max(sub_maxes)
        return absolute_max



    else:

        h1_responses = np.array(list(detector_response(h1, viewingangles = viewingangle).values()))
        h2_responses = np.array(list(detector_response(h2, viewingangles = viewingangle).values()))

        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        pyfftw.interfaces.cache.enable()
        with sp.fft.set_backend(pyfftw.interfaces.scipy_fft):
            with sp.fft.set_workers(os.cpu_count() + 2):
                if not use_gpu:

                    h1resp_f = sp.fft.fft(h1_responses)[:,1:] #drop zero frequency
                    h2resp_f = sp.fft.fft(h2_responses)[:,1:]
                    h2resp_f_star = np.conjugate(h2resp_f)
                    timelength = len(timedomain)
                    DeltaT = timedomain[1]-timedomain[0]
                    frequency_range = sp.fft.fftfreq(int(timelength), d=float(DeltaT))[1:]
                else:
                    h1resp_f = xp.fft.fft(h1_responses)[:,1:] #drop zero frequency
                    h2resp_f = xp.fft.fft(h2_resonses)[:,1:]
                    h2resp_f_star = np.conjugate(h2resp_f)
                    timelength = len(timedomain)
                    DeltaT = timedomain[1]-timedomain[0]
                    frequency_range = xp.fft.fftfreq(int(timelength), d=float(DeltaT))[1:]

                PowerSpectralDensity = LisaSensitivity(np.abs(frequency_range))



                Factor1 = xp.fft.ifft(h1resp_f/PowerSpectralDensity)
                Factor2 = xp.fft.ifft(h2resp_f_star)
                if use_gpu:
                    Factor1 = Factor1.get()
                    Factor2 = Factor2.get()
                convolutions = sp.signal.convolve(Factor1, Factor2, method="fft", mode="full")[[0,-1]].real
                combined_convolutions = np.sum(convolutions,axis=0)
                fullconv = np.max(combined_convolutions)
                res = fullconv

        return np.real(res)

    del convolution, h1,h2,h2_InFreq,h1_InFreq,timelength,DeltaT,frequency_range,frequency_domain,h1f,h2f,h2fstar,PowerSpectralDensity
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return ret


#Naive implementaion of faithfulness (without maximization over time and phase offsets)
def Faithfulness(timedomain, h1, h2,use_gpu=False, data=False, viewingangle = [0,0,0]):
    """
    time domain must be in units of seconds
    """

    if len(h1)!=len(h2):
        warnings.warn("Waveforms have different lengths. Zero padding shorter waveform")
        if len(h1)<len(h2):
            h1 = np.pad(h1, (0,len(h2)-len(h1)))
        elif len(h2)<len(h1):
            h2 = np.pad(h2, (0, len(h1)-len(h2)))
        assert len(h1)==len(h2)

    assert len(timedomain)==len(h1), "time domain has different length than the waveforms. time domain length: {0} waveform 1 length: {1} waveform 2 length: {2}".format(len(timedomain), len(h1), len(h2))


    h1h2 = WaveformInnerProduct(timedomain, h1, h2,use_gpu=use_gpu, maximize=True, viewingangle = viewingangle)
    h1h1 = WaveformInnerProduct(timedomain, h1, h1,use_gpu=use_gpu, viewingangle = viewingangle)
    h2h2 = WaveformInnerProduct(timedomain, h2, h2,use_gpu=use_gpu, viewingangle = viewingangle)
    ret = h1h2/np.sqrt(h1h1*h2h2)

    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if data:
        retdir = {"h1h1":h1h1, "h2h2":h2h2, "h1h2":h1h2,"faithfulness":ret}
        return retdir
    else:
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





##### custom ivp solver
METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}
MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred.",
            2: "Trajectory failed to appreciably change between timesteps."}

def solve_ivp(fun, t_span, y0, method='DOP853', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    """
        See scipy.integrate.solve_ivp docstring
    """
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class."
                         .format(METHODS))

    t0, tf = float(t_span[0]), float(t_span[1])
    y_tol = options.get("ytol", 1e-10)
    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        if np.any(np.abs(ys[-1][0]-y)<y_tol):
            status = 2
            break

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, t_old, t)

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)



    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    else:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    return OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)


















"""


def WaveformInnerProduct(timedomain, h1,h2, fmin=0.0001, fmax=1,use_gpu=False, maximize=False):

    if use_gpu:
        xp = cp
        if isinstance(h1,np.ndarray):
            h1 = xp.asarray(h1)
        if isinstance(h2, np.ndarray):
            h2 = xp.asarray(h2)
    else:
        xp = np

        try:
            if isinstance(h1,cp.ndarray):
                h1 = xp.asarray(h1.get())
        except NameError:
            pass
        try:
            if isinstance(h2,cp.ndarray):
                h2 = xp.asarray(h2.get())
        except NameError:
            pass

    if len(h1)!=len(h2):
        warnings.warn("Waveforms have different lengths. Truncating longer waveform")
        minlength = min([len(h1), len(h2)])-1
        h1 = h1[0:minlength]
        h2 = h2[0:minlength]

    if len(timedomain)!=len(h1) or len(timedomain)!=len(h2):
        raise RuntimeError("time domain has different length than the waveforms")

    if not use_gpu:
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
        pyfftw.interfaces.cache.enable()

        with sp.fft.set_backend(pyfftw.interfaces.scipy_fft):
            h1_InFreq = sp.fft.fft(h1)
            h2_InFreq = sp.fft.fft(h2)
            timelength = len(timedomain)
            DeltaT = timedomain[1]-timedomain[0]
            frequency_range = sp.fft.fftfreq(int(timelength), d=float(DeltaT))
    else:
        h1_InFreq = xp.fft.fft(h1)
        h2_InFreq = xp.fft.fft(h2)
        timelength = len(timedomain)
        DeltaT = timedomain[1]-timedomain[0]
        frequency_range = xp.fft.fftfreq(int(timelength), d=float(DeltaT))

    #Consider real frequencies and ignore zero frequency
    frequency_length = np.argmax(frequency_range)
    frequency_domain = frequency_range[1:frequency_length]
    h1f = h1_InFreq[1:frequency_length]
    h2f = h2_InFreq[1:frequency_length]
    h2fstar = xp.conjugate(h2f)
    PowerSpectralDensity = LisaSensitivity(frequency_domain)


    integrand = h1f*h2fstar/PowerSpectralDensity


    #Assuming frequency domain is monotonically increasing
    mininx = xp.abs(frequency_domain-fmin).argmin()
    maxinx = xp.abs(frequency_domain-fmax).argmin()
    if maxinx==len(frequency_domain)-1:
        maxinx+=1
    masked_frequency_range = frequency_domain[mininx:maxinx]
    masked_integrand = integrand[mininx:maxinx]


    if use_gpu:
        masked_integrand = np.asarray(masked_integrand)
        masked_frequency_range = np.asarray(masked_frequency_range)

    if maximize:
        integral = []
        for i in np.arange(-5,5):
            phase_offset = np.exp(-2*np.pi*1j*masked_frequency_range*i*DeltaT)
            integral.append(sp.integrate.simpson(masked_integrand*phase_offset, x=masked_frequency_range))
        ret = 4*np.real(np.max(integral))

    else:
        ret = 4*sp.integrate.simpson(masked_integrand.real, x=masked_frequency_range)

    del h1,h2,h2_InFreq,h1_InFreq,timelength,DeltaT,frequency_range,frequency_length,frequency_domain,h1f,h2f,h2fstar,PowerSpectralDensity,integrand,mininx,maxinx,masked_frequency_range,masked_integrand
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return ret


"""









"""
if maximize:
    #maximize over coalescence phase and coalescence time
    subresult = []
    Factor1 = xp.fft.ifft(h1f/PowerSpectralDensity)
    Factor2 = xp.fft.ifft(h2fstar)
    if use_gpu:
        convolution = sp.signal.convolve(Factor1.get(), Factor2.get(), method="fft", mode="full")
    else:
        convolution = sp.signal.convolve(Factor1, Factor2, method="fft", mode="full")
    for theta in np.linspace(0,2*np.pi,15):
        resultarray = np.real(convolution*np.exp(1j*theta))
        maxinx = np.argmax(resultarray)
        maxval = resultarray[maxinx]
        subresult.append(maxval)
    ret = max(subresult)
"""
