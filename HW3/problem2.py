# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:45:48 2018

@author: MinF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
def natural_cubic_spline(n, x , a ):

    A = np.zeros(n);
    l = np.zeros(n+1)
    c = np.zeros(n+1)
    z = np.zeros(n+1)
    u = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    # Step 1
    h = (x[n] - x[0])/n
    # Step 2 
    for i in range( 1 , n ):
        A[i] =  3 * (a[i + 1] - a[i]) / h - 3 * (a[i] - a[i - 1]) / h 
    # Step 3 
    l[0] = 1
    u[0] = 0
    z[0] = 0
    #Step 4 
    for i in range( 1 , n ):
        l[i] =  2 * (x[i + 1] - x[i - 1]) - h * u[i - 1] 
        u[i] =  h / l[i]
        z[i] = (A[i] - h * z[i - 1]) / l[i]
    # step 5 
    l[n] = 1
    z[n] = 0
    c[n] = 0
    # Step 6 
    for j in range(n):
        c[n-1-j] = z[n-1-j] - u[n-1-j] * c[n-j];
        b[n-1-j] = (a[n-j] - a[n-1-j]) / h - h * (c[n-j] + 2 * c[n-1-j]) / 3
        d[n-1-j] = (c[n-j] - c[n-1-j]) / (3 * h)
    return b,c,d
def S(x,a,b,c,d,endx):
    return a + b*(x-endx) + c * (x-endx)**2 + d * (x - endx)**3
def f1(x):
    return 1.0 / (1.0 + x**2)
#problem 2
x1 = np.arange(-5,6)
n1 = len(x1)-1
a1 = f1(x1)
xx1 = np.linspace(-5, 5, 100)
b1,c1,d1 = natural_cubic_spline(n1, x1 , a1)
# Plot
fig = plt.figure(1)
axes = fig.add_subplot(1, 1, 1)

axes.plot(x1, a1, 'ko',label="data")
axes.plot(xx1, f1(xx1), 'k',label="True $f(x)$")
for i in range(10):
      xx = np.linspace(x1[i], x1[i+1], 50)
      axes.plot(xx, S(xx,a1[i],b1[i],c1[i],d1[i],x1[i]),label="$S_{%s}(x)$" % i)
axes.set_title(" natural Cubic Splines - f(x) = $1/(1+x^2)$")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim([-6.0, 6.0])
axes.set_ylim([0, 1.5])
axes.legend(loc = 0)
plt.show()