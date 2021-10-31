import scipy.optimize as opt
from scipy.interpolate import lagrange
import numpy as np


###Lagrange Basis function for Trial functions
def Lagr(initial,final,degree,j,x):
    I=np.linspace(initial,final,degree+1)
    v = 1.0
    for k in range(degree+1):
        if k != j:
            v *= (x-I[k]) / (I[j]-I[k])
    return v


###Derivative of Lagrange trial function
def dLagr(initial,final,degree,j,x):
    I=np.linspace(initial,final,degree+1)
    d=0.
    for m in range(degree+1):
        v = 1.0
        for k in range(degree+1):
            if k != j and k !=m:
                v *= (x-I[k]) / (I[j]-I[k])

        if m != j:
            d += (1./(I[j]-I[m]))*v
    return d



###Lagrange basis for Test functions
def TestLagr(initial,final,degree,j,x):
    It=np.linspace(initial,final,degree)
    z= 1.
    if degree > 1:
        for k in range(degree):
            if k != j:
                z *= (x-It[k]) / (It[j]-It[k])
    return z



def evalfunc(Y,t,q,x):
    #evalfunc(vector for numerical solution, mesh for Y, degree of polynomial basis, point to evaluate)
    #Evaluates a piece-wise, degree 'q' interpolation of 'Y' at a point 'x'. 'Y' is known at the nodes of 't'.
    n=len(t)
    ans=0.
    if x == t[-1]:
        ans=Y[-1]
    else:
        
        for i in range(n-1):
            if t[i]<=x<t[i+1]:
                for j in range(q+1):
                    ans+=Y[q*i+j]*Lagr(t[i],t[i+1],q,j,x)
    return(ans)



def evalderiv(Y,t,q,x):
    #evalfunc(vector for numerical solution, mesh for Y, degree of polynomial basis, point to evaluate)
    #Evaluates the derivative of a piece-wise, degree 'q' interpolation of 'Y' at a point 'x'. 'Y' is known at the nodes of 't'.
    n=len(t)
    ans=0.
    for i in range(n-1):
        if t[i]<=x<t[i+1]:
            for j in range(q+1):
                ans+=Y[q*i+j]*dLagr(t[i],t[i+1],q,j,x)
    return(ans)
