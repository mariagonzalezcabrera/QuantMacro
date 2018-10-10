# Problem Set 3 - María González Cabrera

#%% Exercise 2

import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy.optimize as sc
import numpy as np
from scipy.optimize import fsolve
from numpy import random
from numpy import *
from scipy.optimize import *
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution for y0 and epsilons of all the agents of the economy:

random.seed(15)   # Needed for keeping constant the random variables created

y0 = np.random.uniform(0.001, 0.009, 400)

for (i, item) in enumerate(y0):
    if 0.0055<item<0.0087:
        y0[i] = 0.001

y0 = np.array(y0)
sns.distplot(y0, hist=False, rug=True, label='kernel aproximation of y0 distribution')
plt.legend()
        
ey = np.tile(np.array([0.05, -0.05]),200)

# Parameters of the model:
        
n= [1, 1.5, 2.5, 3]     # Permanent productivity
beta= 0.99              # Discount factor
sigma = 3               # Utility parameter 1
k_ = 4                  # Utility parameter 2
v= 4                    # Utility parameter 3
r = 0.618448            # Optimal interest rate obtained after some iterations and different guesses
tau = 0                 # Initial level of proportional labor income taxes
T1 = 0                  # Initial level of lump-sum transfer for first period
T2 = 0                  # Initial level of lump-sum transfer for second period

# Matrix for the eta parameter:
n = np.zeros(400)
n[0:100] = 1
n[100:200] = 1.5
n[200:300] = 2.5
n[300:400] = 3

# Matrix of characteristics:

C = np.append(y0, n)            # C[0]: y0
C = np.append(C, ey)            # C[1]: eta
C = C.reshape(3, 400)           # C[2]: epsilon

# Loop for obtaining the General Equilibrium:

GE = []

for i in range(400):
    def rest(x):
        F =np.zeros(4)
        a = x[0]
        h0 = x[1]
        h1 = x[2]
        lamda = x[3]
        F[0]= np.power((1-tau)*C[1][i]*h0 + C[0][i] + T1 -a, -sigma)*(1-tau)*C[1][i] - k_*np.power(h0,1/v)
        F[1]= beta*np.power((((1-tau)*C[1][i]*h1)+(1+r)*a + T2), -sigma)*(1-tau)*C[1][i] - k_*np.power(h1,1/v)
        F[2]= beta*(np.power(((1-tau)*C[1][i]*h1)+(1+r)*a + T2,-sigma)*(1+r)) - lamda - np.power((1-tau)*C[1][i]*h0 + C[0][i] + T1 -a, -sigma)
        F[3]= ((1-tau)*C[1][i]*h0 + C[0][i] + T2 -a) + (1/(1+r))*((1-tau)*(C[1][i]+C[2][i])*h1 + (1+r)*a + T2) - C[0][i] - (1+r)*(C[1][i]+C[2][i])*h1
        return F
    
    guess= np.array([0.001,0.1,0.1, 1])
    sol = fsolve(rest, guess)
    GE.append(sol)
    GE_sol = np.matrix(GE)
    
np.shape(GE_sol)
GE_sol = np.array(GE_sol)

# Rename variables obtained for the GE and homogeneize their type:
    
a = GE_sol[:,0]
h1 = GE_sol[:,1]
h2 = GE_sol[:,2]

y0 = C[0,:]
ey = C[2,:]

# Consumption today:

c1 = n*h1 + y0 - a

 # Consumption tomorrow:

c2 = (n+ey)*h2 + (1+r)*a 

# Comparative graph for both consumptions:

plt.plot(y0, c1,'.', label = 'Consumption today')
plt.plot(y0, c2,'.', label = 'Consumption tomorrow')
plt.xlim(xmin=0, xmax=0.009)
plt.title('Comparative graph of consumptions')
plt.ylabel('c today and tomorrow')
plt.xlabel('y0')
plt.legend(loc = 'middle right', fontsize = 9)
plt.show()

# PLOT 1:

plt.figure(figsize = (5,15))

plt.subplot(3,1,1)
plt.plot(y0[0:100], a[:100], '.', label = 'eta = 1')
plt.plot(y0[100:200], a[100:200], '.', label = 'eta = 2')
plt.plot(y0[200:300], a[200:300], '.', label = 'eta = 3')
plt.plot(y0[300:400], a[300:400], '.', label = 'eta = 4')
plt.legend()
plt.title('Optimal savings')
plt.ylabel('a')
plt.xlabel('y0')
plt.xlim(xmin=0,xmax=0.009)

plt.subplot(3,1,2)
plt.plot(y0[0:100], c1[:100], '.', label = 'eta = 1')
plt.plot(y0[100:200], c1[100:200], '.', label = 'eta = 2')
plt.plot(y0[200:300], c1[200:300], '.', label = 'eta = 3')
plt.plot(y0[300:400], c1[300:400], '.', label = 'eta = 4')
plt.legend()
plt.title('Consumption today')
plt.ylabel('c today')
plt.xlabel('y0')
plt.xlim(xmin=0,xmax=0.009)

plt.subplot(3,1,3)
plt.plot(y0[0:100], c2[:100], '.', label = 'eta = 1')
plt.plot(y0[100:200], c2[100:200], '.', label = 'eta = 2')
plt.plot(y0[200:300], c2[200:300], '.', label = 'eta = 3')
plt.plot(y0[300:400], c2[300:400], '.', label = 'eta = 4')
plt.legend()
plt.title('Consumption tomorrow')
plt.ylabel('y0')
plt.xlabel('c tomorrow')
plt.xlim(xmin=0,xmax=0.009)

plt.show()

# Saving rate:

sr = a/(y0+n*h1)

# PLOT 2:
    
plt.plot(y0[0:100],sr[0:100], '.',label = 'eta = 1')
plt.plot(y0[100:200],sr[100:200], '.', label = 'eta= 1.5')
plt.plot(y0[200:300],sr[200:300], '.',label = 'eta = 2.5')
plt.plot(y0[300:400],sr[300:400], '.', label = 'eta = 3')
plt.title('Saving rate')
plt.legend(fontsize = 7)
plt.xlabel('y0')
plt.ylabel('sr')
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()


# PLOT 3:

plt.figure(figsize = (5,10))

plt.subplot(2, 1, 1)
plt.plot(y0[0:100],h1[0:100], '.', label = 'eta = 1')
plt.plot(y0[100:200],h1[100:200], '.', label = 'eta = 1.5')
plt.plot(y0[200:300],h1[200:300], '.',label = 'eta = 2.5')
plt.plot(y0[300:400],h1[300:400], '.', label = 'eta = 3')
plt.title('Hours worked today')
plt.legend(fontsize = 7)
plt.xlabel('y0')
plt.ylabel('h1')
plt.xlim(xmin = 0, xmax= 0.009)

plt.subplot(2, 1, 2)
plt.plot(y0[0:100],h2[0:100], '.', label = 'eta = 1')
plt.plot(y0[100:200],h2[100:200], '.', label = 'eta = 1.5')
plt.plot(y0[200:300],h2[200:300], '.',label = 'eta = 2.5')
plt.plot(y0[30:400],h2[300:400], '.', label = 'eta = 3')
plt.title('Hours worked tomorrow')
plt.legend(fontsize = 7)
plt.xlabel('y0')
plt.ylabel('h2')
plt.xlim(xmin = 0, xmax= 0.009)
plt.show()


# Consumption and Income Growths:

cg = (c2-c1)/c1
wh2 = (n+ey)*h2
wh1 = n*h1
ig = (wh2-wh1)/wh1

# Expectation of consumption growth:

exp_c2 = n*h2 + (1+r)*a 
exp_cg = (exp_c2-c1)/c1
exp_ig = (n*h2 - wh1)/wh1

# PLOT 5:

plt.figure(figsize = (5,15))

plt.subplot(2,1,1)
plt.plot(y0[0:100], cg[:100], '.', label = 'cg eta = 1')
plt.plot(y0[100:200], cg[100:200], '.', label = 'cg eta = 2')
plt.plot(y0[200:300], cg[200:300], '.', label = 'cg eta = 3')
plt.plot(y0[300:400], cg[300:400], '.', label = 'cg eta = 4')
plt.plot(y0[0:100], exp_cg[:100], '.', label = 'exp_cg eta = 1')
plt.plot(y0[100:200], exp_cg[100:200], '.', label = 'exp_cg eta = 2')
plt.plot(y0[200:300], exp_cg[200:300], '.', label = 'exp_cg eta = 3')
plt.plot(y0[300:400], exp_cg[300:400], '.', label = 'exp_cg eta = 4')
plt.legend(fontsize = 8)
plt.title('Actual and expected consumption growth')
plt.ylabel('cg and exp_cg')
plt.xlabel('y0')
plt.xlim(xmin=0,xmax=0.009)

plt.subplot(3,1,3)
plt.plot(y0[0:100], ig[:100], '.', label = 'ig eta = 1')
plt.plot(y0[100:200], ig[100:200], '.', label = 'ig eta = 2')
plt.plot(y0[200:300], ig[200:300], '.', label = 'ig eta = 3')
plt.plot(y0[300:400], ig[300:400], '.', label = 'ig eta = 4')
plt.plot(y0[0:100], exp_ig[:100], '.', label = 'exp_ig eta = 1')
plt.plot(y0[100:200], exp_ig[100:200], '.', label = 'exp_ig eta = 2')
plt.plot(y0[200:300], exp_ig[200:300], '.', label = 'exp_ig eta = 3')
plt.plot(y0[300:400], exp_ig[300:400], '.', label = 'exp_ig eta = 4')
plt.legend(fontsize = 8)
plt.title('Actual and expected income growth')
plt.ylabel('ig and exp_ig')
plt.xlabel('y0')
plt.xlim(xmin=0,xmax=0.009)

plt.show()

# Actual and expected elasticity:

actual_elas = cg/ig
exp_elas = exp_cg/exp_ig

plt.plot(y0[0:100], actual_elas[:100], '.', label = 'elas eta = 1')
plt.plot(y0[100:200], actual_elas[100:200], '.', label = 'elas eta = 2')
plt.plot(y0[200:300], actual_elas[200:300], '.', label = 'elas eta = 3')
plt.plot(y0[300:400], actual_elas[300:400], '.', label = 'elas eta = 4')
plt.plot(y0[0:100], exp_elas[:100], '.', label = 'exp_elas eta = 1')
plt.plot(y0[100:200], exp_elas[100:200], '.', label = 'exp_elas eta = 2')
plt.plot(y0[200:300], exp_elas[200:300], '.', label = 'exp_elas eta = 3')
plt.plot(y0[300:400], exp_elas[300:400], '.', label = 'exp_elas eta = 4')
plt.legend(fontsize = 8)
plt.title('Actual and expected elasticity')
plt.ylabel('elas and exp_elas')
plt.xlabel('y0')
plt.xlim(xmin=0,xmax=0.009)
plt.show()

# PLOT 6:

# We iterate with the following range for r: 0.616, 0.617, 0.618448, 0.62, 0.635

S = [1.6690939832866072, 1.763443488952173, 1.8995888673509453, 2.044894404490482, 3.5726217302169485] 
rgrid = [0.616,0.617, 0.618448, 0.62, 0.635]

D = [2.1669721737888206, 2.070025960968001, 1.9301182226722662, 1.7807771178807745, 0.525298445120335]

plt.plot(rgrid, S, rgrid, D)
plt.show()