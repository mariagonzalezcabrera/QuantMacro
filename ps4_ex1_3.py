# Problem Set 4. Quant Macro - María González Cabrera

# In collaboration with Germán Sánchez Arce

# Import packages

import numpy as np
from numpy import vectorize
import timeit
import matplotlib.pyplot as plt
#%% Steady state setting h = 1 -> eliminate the part of the utility with h

# Parametrization of the model:

theeta = 0.679      # labor share
beta = 0.988        # discount factor
delta = 0.013       # depreciation rate


# For computing the steady state we normalize output to one:

y = 1
h = 1
kss = 42.55
iss = delta
css = 1 - delta

#%% CHEBYSHEV

start = timeit.default_timer()

# Discretize the variable of interest:

ki = np.array(np.linspace(0.01, 50, 120))
ki = np.tile(ki, 120)
ki = np.split(ki, 120)
ki = np.transpose(ki)

kj = np.array(np.linspace(0.01, 50, 120))
kj = np.tile(kj, 120)
kj = np.split(kj, 120)

# Define the return matrix M, which give us the utilities for all possible combinations of capital:

@vectorize
def M(ki, kj):
    
         return np.log(pow(ki, 1-theeta) - kj + (1-delta)*ki)
     
M = M(ki, kj)
M = np.nan_to_num(M)
M[M==0] = -1000

# The approximation of the value function using Chebyshev approach is given by:

n = 120
m = n+1 # Chebyshev collocation method
epsilon = 0.001 # Tolerance parameter
k = np.array(np.linspace(0.01, 50, m)) # Array for capital
x = np.polynomial.chebyshev.chebroots(k) # Chebyshev roots for a (-1,1) interval
z = ((0.01+50)/2)+((50-0.01)/2)*x

def V(k):
    return (np.log(pow(k, 1-theeta) - k + (1-delta)*k))

# Setting an initial guess for theeta:

y = V(z)
coef = np.polyfit(z, y, n)
Vs = np.polyval(coef, z) 
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

Vs1 = np.amax(X, axis=1)
    
diffVs =  Vs1 - Vs

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for diffVs in range(1, 80):
    
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    
    X = M + beta*V
    
    Vs1 = np.amax(X, axis=1)
    g = np.argmax(X, axis=1)

    diffVs = Vs1 - Vs
    
    count += 1

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Chebyshev approximation')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()