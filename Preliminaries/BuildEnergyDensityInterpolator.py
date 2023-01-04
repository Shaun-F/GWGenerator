takenumber = 10

from scipy.interpolate import LinearNDInterpolator as linint
import numpy as np
import os, re
import glob
alpha_match_1 = re.compile("Alpha_\d+_\d+")
alpha_match_2 = re.compile("Alpha_\d+")
modeovertonematch = "Mode_1_Overtone_0"
digit_regex = re.compile("\d+")
poly_fit_func = lambda x,a,b,c,d: a + b*x + c*x**2 + d*x**3
notnan = lambda arr: np.logical_not(np.isnan(arr))

#function to extract alpha value
def extract_alpha(string):
    alphastr = alpha_match_1.findall(string)
    if len(alphastr)==0:
        alphastr = alpha_match_2.findall(string)
    if isinstance(alphastr,str):
        pass
    elif isinstance(alphastr,list):
        alphastr=alphastr[0]
    alphapieces = digit_regex.findall(alphastr)
    if len(alphapieces)>1:
        alphavalue = float(alphapieces[0])/float(alphapieces[1])
    else:
        alphavalue = float(alphapieces[0])
    return alphavalue

#import all data
DataPath = os.path.dirname(os.path.abspath(__file__))+"/../ProcaData/"
AllFileNames = glob.glob(DataPath+"BHSpin*")
AllData = [np.load(file) for file in AllFileNames]


#filter data, extracting mode=1 and overtone=0 data
modeovertonebool = [bool(re.search(modeovertonematch, i.fid.match)) for i in AllData]
newalldata = []
for inx,boolval in enumerate(modeovertonebool):
    if boolval:
        newalldata.append(alldata[inx])
alldata = newalldata


#generate intermediate data (alphavalue, minradial, max radial, min theta, max theta, interpolation function)
alphavalues, minmaxradial, minmaxtheta, interpfuncs = [],[],[],[]
for inx,dat in enumerate(alldata):
    raddat = dat["RadialData"]
    thetdat = dat["ThetaData"][0:100]
    nonnanthetdat = notnan(thetdat)
    endat = dat["EnergyData"][:,nonnanthetdat]

    interpfunc = spint.RectBivariateSpline(raddat,thetdat[nonnanthetdat], endat)
    alphavalue = extract_alpha(dat.fid.name)

    alphavalues.append(alphavalue)
    minmaxradial.append([min(raddat), max(raddat)])
    minmaxtheta.append([min(thetdat), max(thetdat)])
    interpfuncs.append(interpfunc)


#sort the lists according to alpha value and take first N alpha values
indexsort = np.argsort(alphavals)
sorted_alphavals = np.asarray(alphavals)[indexsort][:takenumber]
sorted_interpfuncs = np.asarray(interpfuncs)[indexsort][:takenumber]
sorted_maxradial = np.asarray([i[-1] for i in minmaxradial])[indexsort][:takenumber]
sorted_minradial = np.asarray([i[0] for i in minmaxradial])[indexsort][:takenumber]
sorted_maxtheta = np.asarray([i[-1] for i in minmaxtheta])[indexsort][:takenumber]
sorted_mintheta = np.asarray([i[0] for i in minmaxtheta])[indexsort][:takenumber]


maxminradial = max(sorted_minradial)
minmaxradial = min(sorted_maxradial)
maxmintheta = max(sorted_mintheta)
minmaxtheta = min(sorted_maxtheta)

NewRadialDomain = np.arange(maxminradial, minmaxradial, 0.1)
NewThetaDomain = np.arange(maxmintheta, minmaxtheta, 0.1)
new
