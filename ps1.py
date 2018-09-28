# Problem set 1. María González Cabrera

# I did this problem set with the help of Germán Sánchez and Joan Alegre

# Question 1. Function Approximation: Univariate

#%% 1.

# First of all we need to import the packages and functions that we will use:

from sympy import symbols
from sympy import diff
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim

# Taylor approximation of order 1:

x = symbols('x')

a = 1
f = x**0.321
taylor1 = f.subs(x, a)+f.diff(x).subs(x, a)*(x-a)
print('-The first order approximation solution around x = 1 is:', taylor1)

# A more efficient way for calculating a Taylor approximation of order n is given by the following procedure:

def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n-1)

def taylor(f, a, n):
    k = 0
    p = 0
    while k <= n:
        p = p + (f.diff(x, k).subs(x, a))/(factorial(k))*(x-a)**k
        k += 1
    return p

print('-Using the general formula we have the same result for the approximation of order 1 around x = 1:', taylor(f,1,1))

# Now we can use this general formula for the other order approximations of the exercise:

print('-Second order approximation around x = 1', taylor(f, 1, 2))
print('')
print('-Fifth order approximation around x = 1', taylor(f, 1, 5))
print('')
print('-Twentieth order approximation around x = 1', taylor(f, 1, 20))
print('')

# In order to plot the approximations of the real function, we need to change the type of x:

x = np.linspace(0,4,50)

# I know that this is an inneficient way of doing that, but I was trying different ways and nothing.

f0 = x**0.321
t1 = 0.321*x + 0.679
t2 = 0.321*x - 0.1089795*(x - 1)**2 + 0.679
t5 = 0.321*x + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
t20 = 0.321*x - 0.00465389246518441*(x - 1)**20 + 0.00498302100239243*(x - 1)**19 - 0.00535535941204005*(x - 1)**18 + 0.00577951132662155*(x - 1)**17 - 0.00626645146709397*(x - 1)**16 + 0.00683038514023459*(x - 1)**15 - 0.00749000490558658*(x - 1)**14 + 0.0082703737422677*(x - 1)**13 - 0.00920582743809231*(x - 1)**12 + 0.0103445949299661*(x - 1)**11 - 0.0117564360191783*(x - 1)**10 + 0.0135458417089277*(x - 1)**9 - 0.0158761004532294*(x - 1)**8 + 0.0190161406836106*(x - 1)**7 - 0.0234395113198229*(x - 1)**6 + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679

# We want to plot the previous solutions:

plt.plot(x, f0, 'k', label = 'Original function')
plt.plot(x, t1, label = 'Taylor order 1')
plt.plot(x, t2, label = 'Taylor order 2')
plt.plot(x, t5, label = 'Taylor order 5')
plt.plot(x, t20, label = 'Taylor order 20')
plt.legend(loc = 'upper left')
plt.ylim(ymin = -2, ymax = 7)
plt.xlim(xmin = 0, xmax = 4)
plt.title('Taylor approximations', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% 2.

# Taylor approximation around x = 2 with the ramp function

x = symbols('x')

# Our function is the ramp function, but we realized that we only will be focused in a positive range for x, so the analized function is as follows:

f = (2*x)/2
a = 2

# From now on we proceed  in a similar way as in the previous exercise:

def taylor(f, a, n):
    k = 0
    p = 0
    while k <= n:
        p = p + (f.diff(x,k).subs(x,a))/(factorial(k))*(x-a)**k
        k += 1
    return p

print("-Second order Taylor approximation:", taylor(f,2,2))
print("")
print("-Fifth order Taylor approximation:", taylor(f,2,5))
print("")
print("-Twentieth order Taylor approximation:", taylor(f,2,20))
print("")

x=np.linspace(-2, 6, 50)
f0=(x+abs(x))/2
t1 = x
t2 = x
t5 = x
t20 = x

plt.plot(x, f0, 'k', label = 'Original function')
plt.plot(x, t1,'-', label = 'Taylor order 1')
plt.plot(x, t2,'--', label = 'Taylor order 2')
plt.plot(x, t5, label = 'Taylor order 5')
plt.plot(x, t20,'.', label = 'Taylor order 20')
plt.legend(loc = 'upper left')
plt.ylim(ymin = -2, ymax = 7)
plt.xlim(xmin = -2, xmax = 6)
plt.title('Taylor approximations of the Ramp Function', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% 3. Evenly spaced interpolation nodes and a cubic polynomial

# Import packages
from sympy import symbols
from sympy import diff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
import math
from scipy import interpolate
from scipy.interpolate import interp1d

#%% First function:

# Interpolations with monomials:

x = np.linspace(-1, 1, num = 20, endpoint = True) # If we increase the range, we could see better the behaviour of that function
y = np.exp(1/x)

# First part. Interpolations with monomials:

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return np.exp(1/x)

vector = np.linspace(-1, 1, num=20, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2, 5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomial:

def y(x):
    return np.exp(1/x)

vector = np.linspace(-1, 1, num=20, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

# With chebfit we obtain the coefficients of the Chevyshev polynomial, and chebval constructs the polynomial

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 1', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev - Eq. 1', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% Second function:

# First part. Interpolations with monomials:

x = np.linspace(-1, 1, num = 40, endpoint = True)
y = 1/(1+25*x**2)

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, 'o', label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return 1/(1+25*x**2)

vector = np.linspace(-1, 1, num=40, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2 ,5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomial:

def y(x):
    return 1/(1+25*x**2)

vector = np.linspace(-1, 1, num=40, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left', fontsize = 9)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 2', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.ylim(ymin = 0, ymax = 0.55)
plt.title('Errors Chebyshev - Eq. 2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% Third function:

# First part. Interpolations with monomials:

x = np.linspace(-1, 1, num = 40, endpoint = True)
y = (x+abs(x))/2

pol3 = np.polyfit(x, y, 3)
val3 = np.polyval(pol3, x) 

pol5 = np.polyfit(x, y, 5)
val5 = np.polyval(pol5, x)

pol10 = np.polyfit(x, y, 10)
val10 = np.polyval(pol10, x)

plt.plot(x, y, 'o', label = 'Original function')
plt.plot(x, val3,'-', label = 'Interpolation order 3')
plt.plot(x, val5,'--', label = 'Interpolation order 5')
plt.plot(x, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors interpolation with monomials:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(x, error1,'-', label = 'Error of order 1')
plt.plot(x, error3,'--', label = 'Error of order 3')
plt.plot(x, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Second part. Chebyshev approximation with monomials:

def y(x):
    return (x+abs(x))/2

vector = np.linspace(-1, 1, num=40, endpoint=True)

ch = np.polynomial.chebyshev.chebroots(vector)

y2 = y(ch)

pol3 = np.polyfit(ch, y2, 3)
val3 = np.polyval(pol3, ch) 

pol5 = np.polyfit(ch, y2, 5)
val5 = np.polyval(pol5, ch)

pol10 = np.polyfit(ch, y2, 10)
val10 = np.polyval(pol10, ch)

plt.plot(ch, y2,'o', label = 'Original function')
plt.plot(ch, val3,'-', label = 'Interpolation order 3')
plt.plot(ch, val5,'--', label = 'Interpolation order 5')
plt.plot(ch, val10,':', label = 'Interpolation order 10')
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Errors Chebyshev interpolation with monomials:

error1 = abs(y2-val3)
error3 = abs(y2-val5)
error5 = abs(y2-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chebyshev monomial interpolations - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Third part. Chebyshev approximation with Chebyshev polynomials:

def y(x):
    return (x+abs(x))/2

vector = np.linspace(-1, 1, num=40, endpoint=True)
ch = np.polynomial.chebyshev.chebroots(vector)

y = y(ch)
    
ch3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
val3 = np.polynomial.chebyshev.chebval(ch, ch3)

ch5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
val5 = np.polynomial.chebyshev.chebval(ch, ch5)

ch10 = np.polynomial.chebyshev.chebfit(ch, y, 10)
val10 = np.polynomial.chebyshev.chebval(ch, ch10)

plt.plot(ch, y, label = 'Original function')
plt.plot(ch, val3,'-', label = 'Chebyshev order 3')
plt.plot(ch, val5, '--', label = 'Chebyshev order 5')
plt.plot(ch, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Chebyshev approximation - Eq. 3', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors interpolation of Chebyshev:

error1 = abs(y-val3)
error3 = abs(y-val5)
error5 = abs(y-val10)

plt.plot(ch, error1,'-', label = 'Error of order 1')
plt.plot(ch, error3,'--', label = 'Error of order 3')
plt.plot(ch, error5, ':', label = 'Error of order 5')
plt.legend(loc = 'upper left')
plt.xlim(xmin = -1, xmax = 1)
plt.title('Errors Chevyshev - Eq. 3', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% 4.

# Import packages
from sympy import symbols
from sympy import diff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
import math
from math import cos
from math import pi
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d

alpha = 1.0
ro1 = 1/0.2
ro2 = 1/100

def f(x):
    return (np.exp(-alpha*x))/(ro1+ro2**(np.exp(-alpha*x)))
x = np.linspace(0, 10, num = 20, endpoint = True)
y = f(x)
    
ch3 = np.polynomial.chebyshev.chebfit(x, y, 3)
val3 = np.polynomial.chebyshev.chebval(x, ch3)

ch5 = np.polynomial.chebyshev.chebfit(x, y, 5)
val5 = np.polynomial.chebyshev.chebval(x, ch5)

ch10 = np.polynomial.chebyshev.chebfit(x, y, 10)
val10 = np.polynomial.chebyshev.chebval(x, ch10)

plt.plot(x, f(x), '.',label = 'Original function')
plt.plot(x, val3,'-', label = 'Chebyshev order 3')
plt.plot(x, val5, '--', label = 'Chebyshev order 5')
plt.plot(x, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlim(xmin = 0, xmax = 10)
plt.title('Chebyshev approximation. Ro = 1/0.2', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors:
error3 = abs(y-val3)
error5 = abs(y-val5)
error10 = abs(y-val10)

plt.plot(x, error3,'-', label = 'Error of order 3')
plt.plot(x, error5,'--', label = 'Error of order 5')
plt.plot(x, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper right')
plt.xlim(xmin = 0, xmax = 10)
plt.title('Errors. Ro = 1/0.2', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

# Different ro1

alpha = 1.0
ro1 = 1/0.25
ro2 = 1/100

def f(x):
    return (np.exp(-alpha*x))/(ro1+ro2**(np.exp(-alpha*x)))
x = np.linspace(0, 10, num = 20, endpoint = True)
y = f(x)
    
ch3 = np.polynomial.chebyshev.chebfit(x, y, 3)
val3 = np.polynomial.chebyshev.chebval(x, ch3)

ch5 = np.polynomial.chebyshev.chebfit(x, y, 5)
val5 = np.polynomial.chebyshev.chebval(x, ch5)

ch10 = np.polynomial.chebyshev.chebfit(x, y, 10)
val10 = np.polynomial.chebyshev.chebval(x, ch10)

plt.plot(x, y, '.',label = 'Original function')
plt.plot(x, val3,'-', label = 'Chebyshev order 3')
plt.plot(x, val5, '--', label = 'Chebyshev order 5')
plt.plot(x, val10, ':', label = 'Chebyshev order 10')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlim(xmin = 0, xmax = 10)
plt.title('Chebyshev approximation. Ro = 1/0.25', size=15)
plt.ylabel('f(x)', size = 10)
plt.xlabel('x', size = 10)
plt.show()

# Errors:
error3 = abs(y-val3)
error5 = abs(y-val5)
error10 = abs(y-val10)

plt.plot(x, error3,'-', label = 'Error of order 3')
plt.plot(x, error5,'--', label = 'Error of order 5')
plt.plot(x, error10, ':', label = 'Error of order 10')
plt.legend(loc = 'upper right')
plt.xlim(xmin = 0, xmax = 10)
plt.title('Errors. Ro = 1/0.25', size=15)
plt.ylabel('f(x)', size=10)
plt.xlabel('x', size=10)
plt.show()

#%% Question 2. Function Approximation: Multivariate

# Import packages
from sympy import symbols
from sympy import diff
import sympy.solvers
from sympy import solve
from sympy import diff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
import math
import mpmath
from scipy import interpolate
from scipy.interpolate import interp1d

#%% Isoquants:

h = symbols('h')
k = symbols('k')
a = 0.5
s = 0.25

y = ((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))

def f(k,h):
    return ((1-a)*k**((s-1)/s)+a*h**((s-1)/s))**(s/(s-1))

k = np.linspace(0, 10, num = 100, endpoint = True)

# We compute the isoquants manually:
plt.plot(k, ((0.5**3)/2-k**3)**(1/3), label = 'Percentil 5')
plt.plot(k, ((1**3)/2-k**3)**(1/3), label = 'Percentil 10')
plt.plot(k, ((2.5**3)/2-k**3)**(1/3), label = 'Percentil 25')
plt.plot(k, ((5**3)/2-k**3)**(1/3), label = 'Percentil 50')
plt.plot(k, ((7.5**3)/2-k**3)**(1/3), label = 'Percentil 75')
plt.plot(k, ((9**3)/2-k**3)**(1/3), label = 'Percentil 90')
plt.plot(k, ((9.5**3)/2-k**3)**(1/3), label = 'Percentil 95')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlim(xmin = 0, xmax = 10)
plt.ylim(ymin = 0, ymax = 10)
plt.title('Percentils of isoquant 10', size=15)
plt.ylabel('k', size = 10)
plt.xlabel('h', size = 10)
plt.show()