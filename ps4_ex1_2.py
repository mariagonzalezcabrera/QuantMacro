# Problem Set 4. Quant Macro - María González Cabrera

# In collaboration with Germán Sánchez Arce

# Question 1. Adding labor

import numpy as np
from numpy import vectorize
import sympy as sy
import math as mt
import scipy.optimize as sc
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy.sparse as sp


# Parametrization of the model:

theeta = 0.679      # labor share
beta = 0.988        # discount factor
delta = 0.013       # depreciation rate

css=0.8341
iss=0.1659
hss=0.29999
kss=12.765
kappa=5.24
v=2.0

ki = np.array(np.linspace(0.1, 20, 100))
kj = np.array(np.linspace(0.1, 20, 100))
hi = np.array(np.linspace(0.01, 0.6, 100))

from itertools import product
Inputs = list(product(hi, kj,ki))
Inputs =np.array(Inputs)

hi=Inputs[:,0]
kj=Inputs[:,1]
ki=Inputs[:,2]


@vectorize
def M(ki,kj,hi):
         return np.log(pow(ki, 1-theeta)*pow(hi, theeta) - kj + (1-delta)*ki)-kappa*pow(hi,1+1/v)/(1+1/v)
M = M(ki, kj,hi)
M = np.nan_to_num(M)
M[M == 0] = -100000
M=np.split(M,10000)

for i in range(0,10000):
       M[i] =np.reshape(M[i],[1,100])
M=np.reshape(M,[10000,100])   
M=np.transpose(M)    

V=np.zeros([100,10000])

X = M + beta*V
X = np.nan_to_num(X)
X[X == 0.0000] = -100000
Vs1 = np.max(X, axis=1)
V=np.zeros([100,1])

diffVs = Vs1 - V

count = 0

#Loop:

while count <500:
    
    Vs = Vs1
    V=np.tile(Vs,100)
    V=np.reshape(V,[10000,1])
    V=np.tile(V,100)
    V=np.transpose(V)

    X = M + beta*V
    X = np.nan_to_num(X)
    
    Vs1 = np.amax(X, axis=1)
    diffVs = Vs1 - Vs
    
    count = count+ 1

#Plot the capital stock today w.r.t. Value function 
    
ki = np.array(np.linspace(0.1, 20, 100))

plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()
