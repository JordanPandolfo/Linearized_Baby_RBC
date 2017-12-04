"""
Created on Mon Nov  6 12:00:22 2017

@author: Jordan Pandolfo

This code solves for the baby RBC model by linearizing equilibrium conditions
    around the steady state.  The numerical method copies that from Ellen
    McGrattan's online notes found at:

    
    https://docs.google.com/a/umn.edu/viewer?a=v&pid=sites&srcid=dW1uLmVkdXxlbGxlbi1tY2dyYXR0YW58Z3g6ZDkyMmY5ODlhNDllOTli
    
    
The model has a representative firm with Cobb-Douglas utility and a representative
    consumer with log preferences.  Given this, the policy functions for cosumption
    and capital have a closed-form solution:
    
        c(k)  = (1-alpha*beta)*k**alpha
        k'(k) = alpha*beta*k**alpha

The model has two equilibrium conditions:
    
    exp(z)k**alpha-k'-c                = 0    (Market Clearing)
    beta*exp(z')*alpha*k'**(alpha-1)-1 = 0    (Euler Equation)

where z is log productivity, which follows the AR(1) process:
    
    z' = rho*z + epsilon'    
"""

#----------------------#
#                      # 
#   Import Packages    #
#                      #
#----------------------#
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numdifftools as nd
from numpy import linalg as la
from scipy.linalg import eig

#-----------------------#
#                       #
#   Model Parameters    #
#                       #
#-----------------------#
alpha = .3
beta = .95
rho = .9
A_ss = 0

nx, nc, nz = 1,1,1 #Number of state, endogneous and stochastic/exogenous variables
P = rho*np.eye(nx)  #AR 'matrix'
#-----------------------------#
#                             #
#   Solve for Steady State    #
#                             #  
#-----------------------------#

def SS_System(x):
    """
    x[0]: k
    x[1]: c
    """
    
    cond1 = np.exp(A_ss)*x[0]**alpha-x[1]-x[0]
    cond2 = beta*np.exp(A_ss)*alpha*x[0]**(alpha-1)-1
    
    return cond1, cond2

k_ss, c_ss = fsolve(SS_System,[.3,.3])

#Stack steady state values for full equilibrium system
SS = np.hstack(( k_ss,c_ss,k_ss,c_ss,A_ss,A_ss  ))

#-------------------------------------------------#
#                                                 #
#   Solve for Equilibrium Coefficient Matrices    #
#                                                 # 
#-------------------------------------------------#

def Equilibrium_System(x):
    """
    x[0]: k  state,today
    
    x[1]: c  control,today
    
    x[2]: k'  state,tomorrow
    
    x[3]: c'  control,tomorrow
    
    x[4]: z   exog,today
    
    x[5]: z'   exog,tomorrow
    """
    cond1 = np.exp(x[4])*x[0]**alpha - x[1] - x[2]
    cond2 = beta*x[1]*np.exp(x[5])*alpha*x[2]**(alpha-1)-x[3]
    
    return np.array([ cond1, cond2 ])

#Compute Jacobian of system, evaluated at steady state values
Jac = nd.Jacobian(Equilibrium_System)(SS)  


A1 = Jac[:,(nc+nx):2*(nc+nx)]   #Shape: (eqm conds) x (nx+nc = n) where eqm conds = n 
A2 = Jac[:,0:(nc+nx)]    #same
B2 = Jac[:,2*(nc+nx)+nz:2*(nc+nx)+2*nz]   #shape: (eqm conds) x (nz)
B1 = Jac[:,2*(nc+nx):2*(nc+nx)+nz]   #same


#Solve for A and C matrices (state variable part of policy function)

#Solve Generalized Eigenvalue problem
eig_val = eig(A2,-A1)[0]
eig_vec = eig(A2,-A1)[1]


#Sort eigenvalues and eigenvectors by smallest to largest, in absolute value
inside = np.argsort(np.abs(eig_val))[0:nx]
outside =  np.argsort(np.abs(eig_val))[nx:nx+nc]
   
eig_index = np.concatenate(( inside , outside ))

#Rearrange eigenvalue/eigenvectors, accordingly
eig_order = eig_val[eig_index]
eig_vec_order = eig_vec[:,eig_index]


V11 = eig_vec_order[0:nx,0:nx]
V12 = eig_vec_order[0:nx,nx:(nx+nc)]
V21 = eig_vec_order[nx:(nx+nc),0:nx]
V22 = eig_vec_order[nx:(nx+nc),nx:(nx+nx)]
Del1 = np.eye(nx)*eig_order[0:nx]

A = np.dot( np.dot(V11,Del1) , la.inv(V11))
C = np.dot(V21,la.inv(V11))

#Now, solve for B and D matrices (shock part of policy function)
def BD_System(x,A1,A2,B1,C):
    
    nx = np.shape(C)[1]
    nc = np.shape(C)[0]
    nz = np.shape(B1)[1]        
    
    Bx = x[:nx*nz]   #coefficients for B
    Dx = x[-nc*nz:]  #coefficients for D
    
    Bx = np.reshape(Bx,(nx,nz))
    Dx = np.reshape(Dx,(nc,nz))
    
    #first = np.tile(Bx,(2,1))

    Bxp = np.dot(C,Bx)+ np.dot(Dx,P)    
    
    first = np.vstack( (Bx,Bxp) )
    
    zeroed = np.zeros((nx,nz))
    
    second = np.vstack( (zeroed,Dx) )
    
    full = np.dot(A1,first)+np.dot(A2,second)+B1+np.dot(B2,P)
    
    vectorized = np.reshape(full, (nx+nc)*nz  )
    
    return list(vectorized )

inits = list(np.ones(2))

BD_System_final = lambda x: BD_System(x,A1,A2,B1,C)

solution = fsolve(BD_System_final,inits)

B = solution[:nx*nz]
B = np.reshape(B,(nx,nz))
D = solution[-nc*nz:]
D = np.reshape(D,(nc,nz))

#Numerical solutions
G = lambda x,s:  k_ss + np.real(np.dot(A,x-k_ss)+np.dot(B,s-A_ss)) #state equilibrium law of motion
H = lambda x,s:  c_ss + np.real(np.dot(C,x-k_ss)+np.dot(D,s-A_ss)) #control function

#Closed-form solutions
statefun = lambda x: alpha*beta*x**alpha
cfun = lambda x: (1-alpha*beta)*x**alpha


N = 5    #number of simulation periods
numerical_state   = np.zeros((N))
numerical_control = np.zeros((N))

analytic_state    = np.zeros((N))
analytic_control  = np.zeros((N))

numerical_state[0] = analytic_state[0] = k_ss*1.1     #10 percent shock to steady state

for i in range(N):
    if i != N-1:        
        analytic_state[i+1]    =  statefun(analytic_state[i])          #Exact Solution
        analytic_control[i]    =  cfun(analytic_state[i])
        
        numerical_state[i+1] =  np.real(G(numerical_state[i],A_ss))    #Approx solution
        numerical_control[i] =  np.real(H(numerical_state[i],A_ss))


plt.close()  
plt.subplot(2,1,1)
plt.plot(numerical_state,label='approx')
plt.plot(analytic_state,label='exact')
plt.title('State Dynamics')
plt.legend()
plt.subplot(2,1,2)
plt.plot(numerical_control[:-1],label='approx')
plt.plot(analytic_control[:-1],label='exact')
plt.title('Control Dynamics')
plt.legend()
plt.suptitle('Response to 10% Shock to Steady State Capital')
