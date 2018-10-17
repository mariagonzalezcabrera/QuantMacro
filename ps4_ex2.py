#QUESTION 2
#stochastic shocks



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

ki = np.array(np.linspace(0.01, 1.2, 100))
ki = np.tile(ki, 100)
ki = np.split(ki, 100)
ki = np.transpose(ki)

kj = np.array(np.linspace(0.01, 1.2, 100))
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
M1[M1 == 0] = -1000

def M2(ki, kj,z2):
    
         return np.log(pow(z2*ki, 1-theeta) - kj + (1-delta)*ki)
     
M2 = M2(ki, kj,z2)
M2 = np.nan_to_num(M2)
M2[M2 == 0] = -1000
M= np.concatenate([M1,M2])


# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 1.2, 100))

#Since the V is going to be same for every single state (of z, because it will depend on the ki):

def W(ki,z1,z2):
    return 0.4975124*(np.log(pow(z1*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta))+0.502487*(np.log(pow(z2*ki, 1-theeta) - ki + (1-delta)*ki)/(1-beta))
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
kj = np.array(np.linspace(0.1, 1.2, 100))
kj_opt = kj[g[:]]


# Plot the value function:

plt.figure()
plt.plot(ki, Ws1[0:100])
plt.plot(ki,Ws1[100:200])
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

plt.plot(ki,kj[g[:][0:100]])
plt.plot(ki,kj[g[:][100:200]],':')
plt.title('Argmax of the value functions')
plt.ylabel('capital tomorrow')
plt.xlabel('capital today')
plt.legend()
plt.show()

#%% 2.2 Simulate the economy

import statsmodels.api as sm

zhistory=np.random.choice([1.01,1/1.01],size=100,p=[0.497512,1-0.497512])

#Capital
count=0 
ki=0.1
simulation=np.zeros(100)

g_z1=g[0:100]
g_z2=g[100:200]
simulationk=np.zeros(100)
for i in range (0,100):
    
    if zhistory[i]< 1:
        simulation[i]=g_z2[i]
    if zhistory[i]>1:
        simulation[i]=g_z1[i]
        
    simulationk[i] = kj[int(simulation[i])]
  
#Hp filter

cycle, trend = sm.tsa.filters.hpfilter(simulationk, 6.25)

t=np.linspace(0,100,100)
plt.plot(t,simulationk,label="capital")
plt.plot(trend,'_',label='HP filter')
plt.title('Emulation of capital')
plt.ylabel('k')
plt.xlabel('time')
plt.legend()
plt.show()

#Investment
inv=np.zeros(99)

for i in range(0,99):
    inv[i]=simulationk[i+1]-(1-delta)*simulationk[i]
    

cycle, trend = sm.tsa.filters.hpfilter(inv, 6.25)
    
plt.plot(t[0:99],inv,label="investment")
plt.plot(trend,label='HP filter')
plt.title('Emulation of investment')
plt.ylabel('i')
plt.xlabel('time')
plt.legend()
plt.show()

#Ouput
y=np.zeros(100)
for i in range(0,100):
    y[i]=simulationk[i]**(1-theeta)
    
cycle, trend = sm.tsa.filters.hpfilter(y, 6.25)
plt.plot(t[0:100],y,label="output")
plt.plot(trend,'_',label='HP filter')
plt.title('Emulation of output')
plt.ylabel('y')
plt.xlabel('time')
plt.legend()
plt.show()

#Consumption

c=np.zeros(100)
for i in range(0,99):
    c[i]=simulationk[i]**(1-theeta)-inv[i]
    
cycle, trend = sm.tsa.filters.hpfilter(c, 6.25)
plt.plot(t[0:99],c[0:99],label="consumption")
plt.plot(trend, label="HP filter")
plt.title('Emulation of consumption')
plt.ylabel('c')
plt.xlabel('time')
plt.legend()
plt.show()

#%% 2.3



