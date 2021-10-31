import scipy.optimize as opt
from scipy.interpolate import lagrange
import numpy as np
from basis_functions import *
from solvers import *



###Get Error at a time value
def get_err_lin(Y,phi,F,q,q2,ty,tphi):
    #Y = cG(q) solution to forward problem, with grid 'ty'
    #phi = cG(q2) solution to adjoint problem, with grid 'tphi'
    #F comes from RHS of ODE: y'=F(t)y
    #note: This finds the error in Y at time tphi[-1].
    
    
    #Create grid 't' that coincides with the first part of 'ty', ending with t[-1]=tphi[-1].
    if ty[-1]==tphi[-1]:
        t=ty
    else:
        for i in range(len(ty)):
            if ty[i]>tphi[-1]: #find first element ty[i] that is larger than tphi[-1]
                TY=ty[0:i] #truncate 'ty' so that TY[-1] is the largest element smaller than tphi[-1]
                t= TY.tolist() + [tphi[-1]] #tack on tphi[-1] to the new grid
                t=np.array(t)
                break
    
    (p,w) = np.polynomial.legendre.leggauss(5*q) #obtain points p, and weights w, for a high-order gauss-quadrature
    E=0.
    N=len(t)-1
    Eloc=0.*np.array(range(N))
    for n in range(N): #cycle over subintervals of new grid
        g=((t[n+1]-t[n])/2.)*p+(t[n]+t[n+1])/2. #translated points for quadrature
        for k in range(len(g)): #cycle over points in quadrature
            DYL= evalderiv(Y,ty,q,g[k])
            YL= evalfunc(Y,ty,q,g[k])
            PL= evalfunc(phi,tphi,q2,g[k])
            Eloc[n]+=((t[n+1]-t[n])/2.)*w[k]*PL*(-DYL+F(g[k])*YL) #translated quadrature rule for error expression in paper
    E=sum(Eloc)      
    return([E,Eloc,t])
    
    


def window_numtime(Y,t,q,target):
    #Y=vector of functional applied to solution of ODE
    #t=grid
    #q= degree of cG(q)
    #target=threshold value from NS-QoI
    
    #Computes the NS-QoI,Q(Y)=min{t:S(y(t))=target}, for the solution 'Y'.
    #Finds the 4 nodes of 't' surrounding the NS-QoI; tL<t1<Q(Y)<t2<TR
    M=len(t)
    for i in range(M-1):
        if Y[q*i]>target and Y[q*i+1]<target:
            t1=t[i]
            t2=t[i+1]
            Y1=Y[0:q*i+1]
            Y2=Y[0:q*i+q+1]
            N1=i
            N2=i+1
            NL=i-1
            tL=t[i-1]
            YL=Y[0:q*i-q+1]
            NR=i+2
            tR=t[i+2]
            YR=Y[0:q*i+2*q+1]
            break
        elif Y[q*i]<target and Y[q*i+1]>target:
            t1=t[i]
            t2=t[i+1]
            Y1=Y[0:q*i+1]
            Y2=Y[0:q*i+q+1]
            N1=i
            N2=i+1
            NL=i-1
            tL=t[i-1]
            YL=Y[0:q*i-q+1]
            NR=i+2
            tR=t[i+2]
            YR=Y[0:q*i+2*q+1]
            break
    #compute the NS-QoI 'num_time'
    s=(target-Y1[-1])/(Y2[-1]-Y1[-1])
    num_time=s*t2+(1-s)*t1
    
    #NL,N1,N2,NR, are the indices of the grid nodes tL,t1,t2,tR
    #YL=Y(tL), Y1=Y(t1), Y2=Y(t2), YR=Y(tR)
    return([YL,Y1,Y2,YR,NL,N1,N2,NR,tL,t1,t2,tR,num_time])

