import re
import mpmath

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
    return(float(mpmath.ellipk(m)))

#complete elliptic integral of the 2nd kind
def EllipticE(m):
    return(float(mpmath.ellipe(m)))

#complete elliptic integral of the 3rd kind
def EllipticPi(n,m):
    return(float(mpmath.ellippi(n,m)))
