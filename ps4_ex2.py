# Problem Set 4. Quant Macro - María González Cabrera

# In collaboration with Germán Sánchez Arce

# Question 2. Stochastic shocks

import numpy as np
from numpy import vectorize
import sympy as sy
import math as mt
import scipy.optimize as sc
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy.sparse as sp

# Steady state setting h = 1 -> eliminate the part of the utility with h

# Parametrization of the model:

theeta = 0.679      # labor share
beta = 0.988        # discount factor
delta = 0.013       # depreciation rate


# For computing the steady state we normalize output to one:

y = 1
h = 1

z1=1.01
z2=1/1.01
kss1 = (1/1.01)**(1/(1-0.679))#case when z=1.01
kss2= 1.01**(1/(1-0.679)) #case when z=1/0.1
iss1 = kss1-(1-delta)*kss1
iss2= kss2-(1-delta)*kss2
css1 = 1-iss1
css2 = 1-iss2

# Recursive formulation without productivity shocks

# Discretize the variables of interest (k and h):

ki = np.array(np.linspace(0.01, 10, 100))
ki = np.tile(ki, 100)
ki = np.split(ki, 100)
ki = np.transpose(ki)

kj = np.array(np.linspace(0.01, 10, 100))
kj = np.tile(kj, 100)
kj = np.split(kj, 100)

z=np.array([z1,z2])

# where ki is capital of today, and kj is capiptal of tomorrow

# Define the return matrix M, which give us the utilities for all possible combinations of capital:

@vectorize
def M1(ki, kj,z1):
    
         return np.log(pow(z1*ki, 1-theeta) - kj + (1-delta)*ki)
M1=M1(ki, kj,z1)
M1 = np.nan_to_num(M1)
M1[M1 == 0] = -100

def M2(ki, kj,z2):
    
         return np.log(pow(z2*ki, 1-theeta) - kj + (1-delta)*ki)
     
M2 = M2(ki, kj,z2)
M2 = np.nan_to_num(M2)
M2[M2 == 0] = -100
M= np.concatenate([M1,M2])


# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 10, 100))

#def V1(ki,z1):
#    return np.log(pow(z1*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta)
#V1=V1(ki,z1)
#def V2(ki,z2):
#    return np.log(pow(z2*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta)
#V2=V2(ki,z2)

#Since the V is going to be same for every single state (of z, because it will depend on the ki):

def W(ki,z1,z2):
    return 1/2*(np.log(pow(z1*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta))+1/2*(np.log(pow(z2*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta))
Ws=W(ki,z1,z2)

# Compute the matrix X with M and V:


W=np.tile(Ws,2)
W=np.transpose(W)
W=np.reshape(W,[200,1])

X = M + beta*W
X = np.nan_to_num(X)

# Compute a vector with the maximum value for each row of X:

Ws1 = np.max(X, axis=1)
Ws1=np.reshape(Ws1,[200,1])

# Compute the difference between the previous vector and our initial guess of the value function:
Ws=np.tile(Ws,2)
Ws=np.reshape(Ws,[200,1])
diffWs = Ws1 - Ws

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

while count<20:
   
    Ws=np.reshape(Ws1,[2,100])
    Wsz1 = np.tile(Ws[0], 100)
    Wsz1= np.reshape(Wsz1,[100,100])
    Wsz2 = np.tile(Ws[1], 100)
    Wsz2= np.reshape(Wsz2,[100,100])
    W=np.concatenate([Wsz1,Wsz2])
    W = np.array(W)
    
    X = M + beta*W
    X = np.nan_to_num(X)
    Ws1 = np.amax(X, axis=1)
    diffWs = np.reshape(Ws1,[200,1]) - np.reshape(Ws,[200,1])
    
    
    count = count+1
    
# Redefine matrix X with the final value function:

Ws1=np.reshape(Ws1,[2,100])
Wsz1 = np.tile(Ws[0], 100)
Wsz1= np.reshape(Wsz1,[100,100])
Wsz2 = np.tile(Ws[1], 100)
Wsz2= np.reshape(Wsz2,[100,100])
W=np.concatenate([Wsz1,Wsz2])
W = np.array(W)

X = M + beta*W
X = np.nan_to_num(X)

# Now we can obtain the decision rule, which give us column number that
# maximizes row i:
Ws1=np.reshape(Ws1,[200,1])
g = np.argmax(X, axis=1)
kj = np.array(np.linspace(0.1, 2, 100))
kj_opt = kj[g[:]]


# Plot the value function:

plt.figure()
plt.plot(ki, Ws1[0:100])
plt.plot(ki,Ws1[100:200])
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()


