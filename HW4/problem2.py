# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:53:46 2018

@author: MinF
"""

import numpy as np
from scipy.integrate import quad

def f1(x):
    return x**3
def f2(x):
    return np.exp(x)
def f3(x):
    return np.sin(x)

def Simpson_integral(x,f):
    H = (x[1]-x[0])/2.
    ff = H/3.*(f(x[0]) + 4. * f((x[0] + x[1])/2.) + f(x[1]))
    true_ff = quad(f,x[0],x[1])[0]
    error = np.abs(ff-true_ff)
    return ff,error

x1 = np.array([0,1])
x2 = np.array([0,10])
x3 = np.array([0,np.pi])

ff1,error1 = Simpson_integral(x1,f1)
print('the a integral:', ff1, 'the abs error :', error1)

ff2,error2 = Simpson_integral(x1,f2)
print('the b integral:', ff2,'the abs error :', error2)

ff3,error3 = Simpson_integral(x2,f2)
print('the c integral:', ff3,'the abs error :', error3)

ff4,error4 = Simpson_integral(x3,f3)
print('the d integral:', ff4,'the abs error :', error4)

    
