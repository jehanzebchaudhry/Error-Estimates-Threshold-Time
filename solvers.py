import scipy.optimize as opt
from scipy.interpolate import lagrange
import numpy as np
from basis_functions import *


###Performs Gauss Quadrature of degree=degree for integral of 'expression' over interval [inital,final]
def GaussQuad(initial,final,degree,expression):
    (p,w) = np.polynomial.legendre.leggauss(degree)
    C=(final-initial)/2.
    g=C*p+(initial+final)/2.
    ans=0.
    for k in range(len(g)):
        term=expression(g[k])
        ans+=C*w[k]*term
    return ans

###CG(q) Linear-ODE-Solver: y'=F(x)y
def solve_lin(y0,F,t,q):   #solve_lin(initial value, RHS function, grid, cG(q) degree)
    #initialize array of zeros for solution
    N=len(t)-1
    y=0.*np.array(range(0,N*q+1))
    y[0]=y0 #set initial value
    
    (p,w) = np.polynomial.legendre.leggauss(5*q) #obtain points p, and weights w, for a high-order gauss-quadrature
    
    #cycle over subintervals of grid
    for n in range(N):
        #endpoints of subinterval
        initial=t[n]
        final=t[n+1]
        
        #build local matrix for cG(q)
        a=np.zeros([q,q+1])
        
        for i in range(q+1):#cycle through terms of trial function
            for j in range(q):#cycle through terms of test function
                expr=lambda x: TestLagr(initial,final,q,j,x)*(dLagr(initial,final,q,i,x) - F(x)*Lagr(initial,final,q,i,x)) #intregral of this is the (j,i)-th entry of matrix 'a'
                a[j,i]=GaussQuad(initial,final,5*q,expr)
                
        #first column of 'a' corresponds to an already-known parameter, so that column is moved to the RHS of the equation Ax=b
        A=a[0:q,1:q+1] #removing first column of 'a'
        b=-y[n*q]*a[:,0] #adding first column of 'a' to RHS
        
        s=np.linalg.solve(A,b) #Solve As=b
        
        #add local sol'n to global
        y[n*q+1:(n+1)*q+1]=s
        
    return(y)

#One step of C-N: y'=F(t,y)
def CN_step_1D(y_old,t0,t1,F):
    dt=t1-t0
    def func(y_new):
        f=y_new-y_old-(dt/2)*(F(t0,y_old)+F(t1,y_new)) #The CN method is this equation =0
        return f
    Y= opt.broyden1(func,.5,f_tol=1e-14) #nonlinear solver for f=0
    return Y

    
#Iterate CN_step to get values of y at all nodes
def solve_CN_1D(y0,t,F): #solve_CN(initial value, mesh)
    N=len(t)-1
    y=np.zeros(N+1)
    y[0]=y0
    for i in range(N):
        y_old=y[i]
        y[i+1]= CN_step_1D(y_old,t[i],t[i+1],F)
    return y



#perfom one step of the secant root-finding method
def sec_method_step(y0,y1,x0,x1,r):
    Y0=y0-r
    Y1=y1-r
    x2=(x0*Y1-x1*Y0)/(Y1-Y0)
    return x2
    
#perform one step of the inverse-quadratic interpolation root-finding method
def inv_quad_step(y0,y1,y2,x0,x1,x2,r):
    Y0,Y1,Y2=y0-r,y1-r,y2-r
    x3=(x0*Y1*Y2)/((Y0-Y1)*(Y0-Y2))   +   (x1*Y0*Y2)/((Y1-Y0)*(Y1-Y2))   +   (x2*Y1*Y0)/((Y2-Y1)*(Y2-Y0))
    return x3