# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:02:29 2017

@author: NCIMBAY
"""

'''Modeling Univariate Distributions'''


#5.3 Location, Scale, and Shape Parameters
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import random

def PlotBeta1(a,b):
    fig=plt.figure()
    x=np.linspace(st.beta.ppf(0.01,a,b),st.beta.ppf(0.99,a,b),100)    
    plt.plot(x,st.beta.pdf(x,a,b))
    
rv=st.beta(1,1)    
    
parameters=[[4,10],[10,4],[7,7]]

for i in parameters:
    PlotBeta1(*i)


#5.4 Skewness, Kurtosis, and Moments

#5.4.1 The Jarque-Bera Test
'''The Jarque-Bera test tests whether the sample data has the skewness and
kurtosis matching a normal distribution.
'''
x=[random.normalvariate(0,1) for _ in range(10000)]
JB=st.jarque_bera(x)
    
#5.5 Heavy-tailed distributions


''' Exponential and Polynomial Tails'''
'''exp (−|y/θ|α)
f(y) ∼ Ay^(−(a+1)) as y→∞   right polynomial tail,
also as pareto tail
'''



'''t-Distributions'''
x=st.t(2.74)
mean,var,skew,kurt=x.stats(moments='mvsk')
#the same as t.stats(2.74,moments='mvsk')



'''5.5.3 Mixture Models'''
'''mix two normal distribution with probability
Consider a distribution that is 90% N(0, 1) and 10% N(0, 25).
A random variable Y with this distribution can be obtained by generating a
normal random variable X with mean 0 and variance 1 and a uniform(0,1) random
variable U that is independent of X. If U <0.9, then Y = X. If U ≥ 0.9,
then Y = 5X. If an independent sample from this distribution is generated,
then the expected percentage of observations from the N(0, 1) component is
90 %. The actual percentage is random; in fact, it has a Binomial(n, 0.9) distribution,
where n is a sample size. By the law of large numbers, the actual
percentage converges to 90% as n → ∞. This distribution could be used to
model a market that has two regimes, the first being “normal volatility” and
second “high volatility,” with the first regime occurring 90% of the time.
'''

'''5.6 Generalized Error Distributions or GED'''

#very complicated



'''Create skewed from symmetric distributions'''


'''5.9 Maximum Likelihood Estimation'''


'''5.10 Fisher Information and the Central Limit Theorem for MLE
    minus the expected second derivative of the log-likehood
'''



'''5.11 Likehood Ratio Tests'''



















    
    
    