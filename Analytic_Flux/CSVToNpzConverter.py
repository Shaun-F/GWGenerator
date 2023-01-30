import numpy as np
import glob
import os,sys
import pandas as pd
import re

curr_path = sys.path[0]
OutFolder = curr_path+"/ProcaEnDenData/"
InFolder = curr_path+"/CSVData/"

if not os.path.exists(OutFolder):
    os.mkdir(OutFolder)
def contains_str(str, regex):
    _regex = re.compile(regex)
    return bool(_regex.search(str))


allfiles = glob.glob(InFolder+"*")

print("*******************\n")
print("Output Folder: {0}".format(OutFolder))
print("Input Folder: {0}".format(InFolder))
print("Number of Input Files: {0}".format(len(allfiles)))
print("\n*******************\n")


allfilenames = np.array(list(map(os.path.basename, allfiles)))
fun = lambda str: contains_str(str, "EnergyDensityCOORDS*")
COORDS_filenames = allfilenames[list(map(fun, allfilenames))]
fun = lambda str: contains_str(str, "EnergyDensityVALUES*")
VALUES_filenames = allfilenames[list(map(fun, allfilenames))]


coordvaluepair = []
for valuestr in VALUES_filenames:
    try:
        data_name = valuestr[20:]
        truth = [i[20:]==valuestr[20:] for i in COORDS_filenames]
        coordstr = COORDS_filenames[truth][0]
        coord_data = pd.read_csv(InFolder+coordstr, header=None, na_values="None")
        value_data = pd.read_csv(InFolder+valuestr, header=None, na_values="None")
        rdata = coord_data[0].dropna().values
        thdata = coord_data[1].dropna().values
        valuedata = value_data.values

        out_filename = OutFolder+data_name[:-4]
        np.savez(out_filename, RadialData=rdata, ThetaData=thdata, EnergyData=valuedata)
    except KeyError:
        print("Error at: \n\t {0} \n\t {1}\n skipping \n".format(valuestr,coordstr))
