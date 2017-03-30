# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:25:22 2017

@author: NCIMBAY
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def BondValue(c,T,r,par):
    """   
    Computers bv=bond value(current prices) corresponding 
    to all values of yield to maturity in the input vector r
    
    INPUT
    c = coupon payment (semiannual)
    T = time to maturity (in years)
    r = vector fo yields to maturity
    par = par value               
    """    

    bv= c/r +(par-c/r)*(1+r)**(-2*T)
    return bv
    
    
r=np.arange(0.02,0.05,0.0001)

price= 1200
C=40
T=30
par=1000

value=BondValue(C,T,r,par)
yield2M=UnivariateSpline(value,r)
figure=plt.figure()
plt.xlabel('yield to maturity')
plt.ylabel('price of bond')
plt.plot(r,value,label='par=1000, coupon payment=40, T=30')
plt.scatter(yield2M(1200),1200,color='r')
plt.legend()


# find the uniroot function in python  to get the root

#Graphing Yield Curves

mk_maturity=pd.read_csv('G:\\书籍\\面试答疑\\Python Lab for Statistics and data analysis\\datasets\\mk_maturity.csv')
mk_zero2=pd.read_csv('G:\\书籍\\面试答疑\\Python Lab for Statistics and data analysis\\datasets\\mk_zero2.csv')
figure2=plt.figure()
plt.xlabel('maturity')
plt.ylabel('yield')
plt.plot(mk_maturity['Maturity'],mk_zero2.ix[5,1:56],label='1985-12-01')
plt.plot(mk_maturity['Maturity'],mk_zero2.ix[6,1:56],color='r',label='1986-01-01')
plt.plot(mk_maturity['Maturity'],mk_zero2.ix[7,1:56],color='y',label="1986-02-01")
plt.plot(mk_maturity['Maturity'],mk_zero2.ix[8,1:56],color='g',label="1986-03-01")
plt.legend(loc=4)


#find a way to give a string vector to the line vector, which will make it easier and more standard

intForward=np.array(mk_maturity['Maturity'])*np.array(mk_zero2.ix[6,1:56])
plt.plot(mk_maturity['Maturity'],intForward)
xout=np.arange(0,20,0.1)
z1=UnivariateSpline(np.array(mk_maturity['Maturity'])[:-7],intForward[:-7])
f=pd.Series(z1(xout))
forward=f.diff()/0.01
plt.plot(xout,forward)