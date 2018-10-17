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

#%% A) BRUTE FORCE ITERATIONS

start = timeit.default_timer()

# Discretize the variable of interest:

ki = np.array(np.linspace(0.01, 50, 120))
ki = np.tile(ki, 120)
ki = np.split(ki, 120)
ki = np.transpose(ki)

kj = np.array(np.linspace(0.01, 50, 120))
kj = np.tile(kj, 120)
kj = np.split(kj, 120)

# where ki is capital of today, and kj is capiptal of tomorrow

# Define the return matrix M, which give us the utilities for all possible combinations of capital:

@vectorize
def M(ki, kj):
    
         return np.log(pow(ki, 1-theeta) - kj + (1-delta)*ki)
     
M = M(ki, kj)
M = np.nan_to_num(M)
M[M==0] = -1000

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V:

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

# Compute a vector with the maximum value for each row of X:

Vs1 = np.max(X, axis=1)

# Compute the difference between the previous vector and our initial guess of the value function:

diffVs = Vs1 - Vs

count = 0
# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for diffVs in range(1, 80):
   
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    
    X = M + beta*V
    
    Vs1 = np.amax(X, axis=1)
    diffVs = Vs1 - Vs
    
    count += 1
    
# Redefine matrix X with the final value function:

X = M + beta*V
X = np.nan_to_num(X)

# Now we can obtain the decision rule, which give us column number that
# maximizes row i:

g = np.argmax(X, axis=1)

kj = np.array(np.linspace(0.01, 50, 120))
kj_opt = kj[g[:]]

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% B) MONOTONICITY OF THE OPTIMAL DECISION RULE:

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

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V (first iteration):

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

Vs1 = np.amax(X, axis = 1)
diffVs = Vs1 - Vs

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for diffVs in range(1, 80):
    
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    g = np.zeros(120)
    
    Vs1 = np.zeros(120)
    
    X = M + beta*V
    
    Vs1[0] = np.amax(X[0])
    g[0] = np.argmax(X[0])
    
    for i in range(1, 120):
       
        g[i] = np.argmax(X[i, int(g[i-1]):-1])
        Vs1[i] = np.amax(X[i, int(g[i-1]):-1])
  
    diffVs = Vs1 - Vs
    
    count += 1

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)
        
# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% C) CONCAVITY OF THE VALUE FUNCTION:

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
M[M == 0] = -1000

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V (first iteration):

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

Vs1 = np.amax(X, axis = 1)
diffVs = Vs1 - Vs

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for diffVs in range(1, 80):
    
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    
    for i in range(0, 119):
        for j in range(1, 120):
                        
            X[0, 0] = M[0,0] + beta*V[0,0]
            X[i, 0] = M[i, 0] + beta*V[i, 0]
            
            Vs1[0] = np.amax(X[0])
            
        if  X[i, j-1] > X[i, j]:
                
            Vs1[i] = X[i, (j-1)]
                
        break
            
    diffVs = Vs1 - Vs

    count += 1
 
stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% LOCAL SEARCH ON THE DECISION RULE:

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

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Fix s_low and s_large:

s_low = 5
s_large = 5

# Compute the matrix X with M and V (first iteration):

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

Vs1 = np.amax(X, axis = 1)
g = np.argmax(X, axis=1)

diffVs = Vs1 - Vs

count = 0

for diffVs in range(1, 80):
    
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)

    X = np.empty((120, 120))
    j_low = np.empty(120)
    j_large = np.empty(120)
    
    for i in range(0, 120):

        j_low[i] = np.amax((1, int(g[i]-s_low)), axis=0)
        j_large[i] = np.min((120, int(g[i]+s_large)), axis=0)
        
        for i in range(0, 120):
            
            for j in range(int(j_low[i]), int(j_large[i])):
                
                X[i, j] = M[i, j] + beta*V[i, j]
            
        Vs1[i] = np.amax(X[i])
        g[i] = np.argmax(X[i])
        
    diffVs = Vs1 - Vs

    count += 1

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% MONOTONICITY AND CONCAVITY OF THE DECISION RULE:

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

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V (first iteration):

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X = M + beta*V

Vs1 = np.amax(X, axis = 1)
diffVs = Vs1 - Vs

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for diffVs in range(1, 80):
    
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    g = np.zeros(120)
    
    Vs1 = np.zeros(120)
    
    X = M + beta*V
    
    Vs1[0] = np.amax(X[0])
    g[0] = np.argmax(X[0])
    
    for i in range(1, 120):
        for j in range(1, 120):
            
            if X[i, j-1] > X[i, j]:
                
                X[i, j] = -10000     # For not taking into account the numbers larger than X[i, j-1] -> concavity
            
        g[i] = np.argmax(X[i, int(g[i-1]):-1])
        Vs1[i] = np.amax(X[i, int(g[i-1]):-1])
            
                            
    diffVs = Vs1 - Vs
    
    count += 1

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% F) HOWARD'S POLICY ITERATION

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

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V:

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X1 = M + beta*V

# Compute a vector with the maximum value for each row of X:

Vs1 = np.max(X1, axis=1)

# Compute the difference between the previous vector and our initial guess of the value function:

g = np.argmax(X1, axis=1)

diffVs = Vs1 - Vs

count = 0
# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:


for diffVs in range(1, 80):
   
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    
    X = M + beta*V   
    
    for i in range(0, 120):
        
        X[i, g] = M[i, g] + beta*V[i, g]
        
        Vs1 = np.amax(X, axis = 1)
    
    diffVs = Vs1 - Vs
    
    count += 1
    
# Redefine matrix X with the final value function:

X = M + beta*V
X = np.nan_to_num(X)

# Now we can obtain the decision rule, which give us column number that
# maximizes row i:

kj = np.array(np.linspace(0.01, 50, 120))
kj_opt = kj[g[:]]

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

#%% G) POLICY ITERATIONS WITH DIFFERENT REASSESSMENTS:

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

# Define our initial guess for the value function V:

ki = np.array(np.linspace(0.01, 50, 120))

def V(ki):
    return (np.log(pow(ki, 1-theeta) - ki + (1-delta)*ki))/(1-beta)

# Compute the matrix X with M and V:

Vs = V(ki)
V = np.tile(Vs, 120)
V = np.split(V, 120)
V = np.array(V)

X1 = M + beta*V

# Compute a vector with the maximum value for each row of X:

Vs1 = np.max(X1, axis=1)

# Compute the difference between the previous vector and our initial guess of the value function:

g = np.argmax(X1, axis=1)

diffVs = Vs1 - Vs

count = 0

# If differences are larger than 1, we iterate taking as new value functions Vs1 up to obtain convergence:

for count in range(100):
   
    Vs = np.transpose(Vs1)
    V = np.tile(Vs, 120)
    V = np.split(V, 120)
    V = np.array(V)
    
    # X1 = M + beta*V   #-> The result doesn't change if we modify g at every step or if we use always the same g associated with the first X (that we have called X1)
    
    X1 = np.zeros((120, 120))
    
    for i in range(0, 120):
        
        X1[i, g] = M[i, g] + beta*V[i, g]
        
        g1 = np.argmax(X1, axis = 1)
        
        for count in range(1, 5):
            
            X1[i, g1] = M[i, g1] + beta*V[i, g1]
            
            g1 = np.argmax(X1, axis = 1)

        for count in range(5, 10):
                                
            X1[i, g1] = M[i, g1] + beta*V[i, g1]
            
            g1 = np.argmax(X1, axis = 1)
            
        for count in range(10, 20):
                                
            X1[i, g1] = M[i, g1] + beta*V[i, g1]
            
            g1 = np.argmax(X1, axis = 1)
                     
        for count in range(20, 50):
            
            X1[i, g1] = M[i, g1] + beta*V[i, g1]
            
            g1 = np.argmax(X1, axis = 1)
            
            Vs1 = np.amax(X1, axis = 1)
#        
    diffVs = Vs1 - Vs
    
    count += 1
    
# Redefine matrix X with the final value function:

X = M + beta*V
X = np.nan_to_num(X)

# Now we can obtain the decision rule, which give us column number that
# maximizes row i:

kj = np.array(np.linspace(0.01, 50, 120))
kj_opt = kj[g[:]]

stop = timeit.default_timer()
execution_time = stop - start
execution_time = round(execution_time, 2)
print('Program executed in', execution_time)

# Plot the value function:
    
plt.figure()
plt.plot(ki, Vs1)
plt.title('Value Function Iteration')
plt.ylabel('Value Function')
plt.xlabel('Capital stock of today')
plt.show()

