

cdef extern from "dIdt8H_5PNe10.h":
	double dEdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne)
	double dLdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne)

def pydEdt(q,p,e,Y,nv,ne):
	return dEdt8H_5PNe10(q,e,p,Y,nv,ne)
def pydLdt(q,p,e,Y,nv,ne):
	return dLdt8H_5PNe10(q,e,p,Y,nv,ne)
