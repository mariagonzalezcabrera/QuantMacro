# PS3 - María González Cabrera

# Packages
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

#%% Exercice 1. a) and b):
# Let's find Z:
y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)
#Delta (depreciation) is clearly 1/16.
d = 1/16 # depreciation
h2z = pow((2*z*ht),0.67) #new worker in efficient terms.
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*h2z+1-d-(1/b)
kguess = 1
k = sc.fsolve(SS,kguess)
print('9.6815 is the new stationary capital when 2 times z')
#%%Exercice 1. c))
def EE(k1,k2,k3):
    return pow(k2,0.33)*h2z-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*h2z+(1-d))*(pow(k1,0.33)*h2z-k2+(1-d)*k1)
#def transv(k1,k2):
#    return pow(b,500)*(0.33*pow(k1,-0.67)*h2z+(1-d))*k1*pow(pow(k1,0.33)*h2z-k2+(1-d)*k1,-1)
K = 9.68
def transition(z): 
    F = np.zeros(200)
    z = z
    F[0] = EE(4,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[199] = 9.68
    F[198] = EE(z[197], z[198], z[199])
    for i in range(1,198):
        F[i] = EE(z[i],z[i+1],z[i+2])
    return F
z = np.ones(200)*4
k = sc.fsolve(transition, z)
k[0] = 4
# I create the domain to plot everything.
kplot = k[0:100]
t = np.linspace(0,100,100)

# I create savings, output and consumption:
yt = pow(kplot,0.33)*h2z
kt2 = k[1:101]
st = kt2-(1-d)*kplot
ct = yt-st

plt.plot(t,kplot, label='capital')
plt.legend()
plt.title('Transition of K from  first S.S to second S.S, first 100 times', size=20)
plt.xlabel('Time')
plt.ylabel('capital')
plt.show()

plt.plot(t,yt, label='Yt output')
plt.plot(t,st, label='st savings')
plt.plot(t,ct, label='ct consumption')
plt.legend(loc='upper right')
plt.title('Transition of the economy', size=20)
plt.xlabel('Time', size = 20)
plt.ylabel('Quantity', size = 20)
plt.show()
#%% Exercice1. d):

y = 1
ht = (31/100)
b = 0.99 # beta
def zeta(z):
    ht = (31/100)
    k = 4
    z = z
    f = pow(k,0.33)*pow(z*ht,0.67)-1
    return f
zguess = 1
z = sc.fsolve(zeta, zguess)

#new hz
hz = pow((z*ht),0.67)

# New euler equation:
def EE2(k1,k2,k3):
    return pow(k2,0.33)*hz-k3+(1-d)*k2-b*(0.33*pow(k2,-0.67)*hz+(1-d))*(pow(k1,0.33)*hz-k2+(1-d)*k1)

k10 = k[9]
# I compute the new Stady stationary:
def SS(k):
    k = k
    return 0.33*pow(k,-0.67)*hz+1-d-(1/b)
kguess = 1
kss = sc.fsolve(SS,kguess)

def transition(z): 
    F = np.zeros(100)
    z = z
    F[0] = EE2(6.801,z[1],z[2])
#    F[499] = transv(z[497],z[498])
    z[99] = 4.84
    F[98] = EE2(z[97], z[98], z[99])
    for i in range(1,98):
        F[i] = EE2(z[i],z[i+1],z[i+2])
    return F
z = np.ones(100)*4
k2 = sc.fsolve(transition, z)
k2[0] = 6.801

#lets plot everything:
kplot = k[0:100]
kfin = np.append(k[0:10],k2[0:90]) 
t = np.linspace(0,100,100)
plt.plot(t,kplot,'--', label='expected transition')
plt.plot(t,kfin, label='actual transition')
plt.axvline(x=10, color='black')
plt.legend()
plt.title('Difference of economy by shock at t=10', size=20)
plt.xlabel('Time', size=20)
plt.ylabel('Capital', size = 20)
plt.show()

#%% Question 2:
y0 = np.random.uniform(0.001,0.009,400)

for (i, item) in enumerate(y0):
    if 0.0055<item<0.0087:        
        y0[i] = 0.001        
y0 = np.array(y0)
sns.distplot(y0, hist=False, rug=True, label='kernel aproximation of y0 distribution')
plt.legend();

def sum(x):
    z = 0
    for i in x:
        z = z + i
    return z

r = 0.618448
sigma = 3
kappa = 4
nu = 4
beta = 0.99
tau = 0
T0 = 0
T1 = 0
          
# I create the matrix of attributes of individuals:
# matrix of NHU valures.
NHU = np.zeros(400)
NHU[0:100] = 1
NHU[100:200] = 1.5
NHU[200:300] = 2.5
NHU[300:400] = 3

# Matrix of epsilons:
EPS = np.tile(np.array([0.05, -0.05]),200)

# I create matrix caracteristics, C:
C = np.append(y0,NHU)
C = np.append(C,EPS)
C = C.reshape(3,400)

Equilibrium = []
  
for i in range(400):
    def rest(x):
        F =np.zeros(4)
        a = x[0]
        h0 = x[1]
        h1 = x[2]
        lamda = x[3]
        F[0]= np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)*(1-tau)*C[1][i] - kappa*np.power(h0,1/nu)
        F[1]= beta*np.power((((1-tau)*C[1][i]*h1)+(1+r)*a + T1), -sigma)*(1-tau)*C[1][i] - kappa*np.power(h1,1/nu)
        F[2]= beta*(np.power(((1-tau)*C[1][i]*h1)+(1+r)*a + T1,-sigma)*(1+r)) - lamda - np.power((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a, -sigma)
        F[3]= ((1-tau)*C[1][i]*h0 + C[0][i] + T0 -a) + (1/(1+r))*((1-tau)*(C[1][i]+C[2][i])*h1 + (1+r)*a + T1) - C[0][i] - (1+r)*(C[1][i]+C[2][i])*h1
        return F
    sguess= np.array([0.001,0.1,0.1, 1])
    sol = fsolve(rest,sguess)
    Equilibrium.append(sol)
    eq_mat = np.matrix(Equilibrium)
shape(eq_mat)
eq_mat = np.array(eq_mat)

a = eq_mat[0]
h0 = eq_mat[1]
h1 = eq_mat[2]
# Now I will find consumptions:
C1 = np.zeros(400)
C2 = np.zeros(400)
#for i in range(400):
#    C1[i] = (1-tau)*C[1][i]*h0[i] + C[0][i] + T0 -a[i]
#    C2[i] = (1-tau)*(C[1][i]+C[2][i])*h1[i]+(1+r)*a[i]+T1 

plt.scatter(y0,C1, label = 'consumption 1st period')
plt.scatter(y0,C2, label = 'consumption 2nd period')
plt.scatter(y0,h0, label = '')
plt.show()