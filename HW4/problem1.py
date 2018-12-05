# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.abc import x,y

#x,y = symbols("x y", positive=True)
def Guass_appr(n,a,b,case):  
    A,B,N = symbols("A B N", positive=True)
    coeff1 = sqrt((2*N+1)/2)  * 1/(2**N * factorial(N))* sqrt(2/(B-A))
    values = {A: a, B: b, N: n}
    y = Rational(2,(b-a)) * x - Rational(2*a,(b-a)) - 1
    ff = (x**2-1)**n
    dff = diff(ff,x,n)
    dff = dff.subs(x,y)
    q = simplify( coeff1.subs(values)*dff)
    print ('q_'+ str(n) +':' ,q)
    if case == 0:
       f = x**2
    elif case == 1:
       f = x**(Rational(3,2))
    else:
       f = (1 + x**2)**(-1)
    coef = integrate(q*f,(x,a,b))
    print('c_' + str(n) +':', coef)
    return coef,q


# (a)
print ('solution of question 1(a):')
case = 0
aa = 0
ba = 1
na = 1
ca = np.zeros([na+1,1])
qa = [0]*3
Pa = 0
for i in range(na+1):
    ca,qa= Guass_appr(i,aa,ba,case) 
    Pa = Pa + ca * qa
Paa = lambdify(x, Pa)
xa = np.linspace(0, 1, 100)
# Plot
fig = plt.figure(1)
axes = fig.add_subplot(1, 1, 1)  
axes.plot(xa,Paa(xa), label="polynomial $p_1$")
axes.plot(xa, xa**2, label="True $f(x)$")
axes.set_title(" f(x) = $x^2$")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim([0, 1.0])
axes.set_ylim([-0.3, 1])
axes.legend(loc = 0)
plt.show()


#(b)
print ('solution of question 1(b):')
case_b = 1
ab = 0
bb = 1
nb = 2
Pb = 0
for i in range(nb+1):
    cb,qb= Guass_appr(i,ab,bb,case_b) 
    Pb = Pb + cb * qb
Pbb = lambdify(x, Pb)
xb = np.linspace(0, 1, 100)
# Plot
fig = plt.figure(2)
axes = fig.add_subplot(1, 1, 1)  
axes.plot(xb,Pbb(xb), label="polynomial $p_2$")
axes.plot(xb, xb**(3/2),label="True $f(x)$")
axes.set_title(r" f(x) = $x^{3/2}$")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim([0, 1.0])
axes.set_ylim([-0.1, 1])
axes.legend(loc = 0)
plt.show()

#(c)
print ('solution of question 1(c):')
case_c = 2
ac = -5
bc = 5
nc = 8
Pc = 0
for i in range(nc+1):
    cc,qc= Guass_appr(i,ac,bc,case_c) 
    Pc = Pc + cc * qc
Pcc = lambdify(x, Pc)
xc = np.linspace(-5, 5, 100)
# Plot
fig = plt.figure(3)
axes = fig.add_subplot(1, 1, 1)  
axes.plot(xc,Pcc(xc), label="polynomial $p_8$")
axes.plot(xc, 1/(1+xc**2),label="True $f(x)$")
axes.set_title(" f(x) = $1/(1 + x^2)$")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1])
axes.legend(loc = 0)
plt.show()
