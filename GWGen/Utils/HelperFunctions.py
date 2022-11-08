import re
from mpmath import *
mp.dps=25
mp.pretty=True

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
