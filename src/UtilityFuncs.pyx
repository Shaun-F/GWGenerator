cdef extern from "Utility.hh":
    void KerrGeoCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double a, double p, double e, double x);


def pyKerrGeoCoordinateFrequencies(double a, double p, double e, double x):

    cdef double OmegaPhi = 0.0
    cdef double OmegaTheta = 0.0
    cdef double OmegaR = 0.0

    KerrGeoCoordinateFrequencies(&OmegaPhi, &OmegaTheta, &OmegaR, a, p, e, x)

    return (OmegaPhi, OmegaTheta, OmegaR)
