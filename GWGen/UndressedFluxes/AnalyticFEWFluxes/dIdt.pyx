

cdef extern from "dIdt8H_5PNe10.h":
	double dEdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne)
	double dLdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne)
	double dCdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
	double dpdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
	double dedt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
	double dYdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);


def pydEdt(q,e,p,Y,nv,ne):
	return dEdt8H_5PNe10(q,p,e,Y,nv,ne)
def pydLdt(q,e,p,Y,nv,ne):
	return dLdt8H_5PNe10(q,p,e,Y,nv,ne)
def pydCdt(q,e,p,Y,nv,ne):
	return dCdt8H_5PNe10(q,p,e,Y,nv,ne)
def pydpdt(q,e,p,Y,nv,ne):
	return dpdt8H_5PNe10(q,p,e,Y,nv,ne)
def pydedt(q,e,p,Y,nv,ne):
	return dedt8H_5PNe10(q,p,e,Y,nv,ne)
def pydYdt(q,e,p,Y,nv,ne):
	return dYdt8H_5PNe10(q,p,e,Y,nv,ne)
