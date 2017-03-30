# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:34:20 2017

@author: NCIMBAY
"""

import pandas_datareader as pdr
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

Data=pdr.get_data_yahoo('^GSPC','1/1/1981','12/30/1991',interval='d')['Close']
daily_return=Data.diff()/Data.shift(1)
fig1=plt.figure()
plt.plot(daily_return,color='black')


#kernel density estimator
daily_return=daily_return.dropna()
kernel1=st.gaussian_kde(daily_return,bw_method=3)
kernel2=st.gaussian_kde(daily_return,bw_method=1)
kernel3=st.gaussian_kde(daily_return,bw_method=1/10)
x=np.arange(-0.04,0.04,0.001)
plt.plot(x,kernel1(x))
plt.plot(x,kernel2(x),linestyle=':')
plt.plot(x,kernel3(x),linestyle='--')


'''
Though simple to compute, the KDE has some problems. In particular, it
is often too bumpy in the tails.
'''

#Order Statistics, the Sample CDF, and Sample Quantiles
random.seed(991155)
x=np.array([random.normalvariate(0,1) for _ in range(150)])
edcf_norm=ECDF(x)
tt=np.arange(-3,3,0.01)
fig2=plt.figure()
plt.plot(tt,edcf_norm(tt),label='EDF')
plt.plot(tt,st.norm.cdf(tt),color='r',linestyle='--',label='CDF')
plt.legend(loc=4)


# 4.3.2 Normal Probability Plots
'''
Therefore, except for sampling variation, a
plot of the sample quantiles versus Φ−1 will be linear. One version of the
normal probability plot is a plot of Y(i) versus Φ−1{(i − 1/2)/n}.
'''

N=[20,150,1000]
for n in N:
    x=np.array([random.normalvariate(0,1) for _ in range(n)])
    x=np.array(sorted(x))
    mean=x.mean()
    sd=x.std()
    a=[mean+sd*norm.ppf((i-1/2)/n) for i in range(1,n+1)]
    b=[x[i] for i in range(n)]
    plt.subplot(2,2,N.index(n)+1)
    plt.scatter(b,a)
    plt.plot(np.arange(-4,4),np.arange(-4,4))


#4.3.3 Half-Normal Plots
'''
The half-normal plot is a variation of the normal plot used for detecting
outlying data rather than checking for a normal distribution.
For example,
suppose one has data Y1, . . . , Yn and wants to see whether any of the absolute
deviations |Y1 −Y |, . . . , |Yn −Y | from the mean are unusual. In a half-normal
plot, these deviation are plotted against the quantiles of |Z|, where Z is N(0, 1)
distributed. More precisely, a half-normal plot is a scatterplot of the order
statistics of the absolute values of the data against Φ−1{(n + i)/(2n + 1)},
'''
N=[20,150,1000]
for n in N:
    x=np.array([random.normalvariate(0,1) for _ in range(n)])
    mean=x.mean()
    sd=x.std()
    a=sorted([abs(y-mean) for y in x])
    b=[norm.ppf((n+i)/(2*n+1)) for i in range(1,n+1)]
    plt.subplot(2,2,N.index(n)+1)
    plt.scatter(a,b)
    plt.plot(np.arange(-4,4),np.arange(-4,4))


# Quantile-Quantile Plots
'''
Normal probability plots are special cases of quantile-quantile plots, also
known as QQ plots. A QQ plot is a plot of the quantiles of one sample or
distribution against the quantiles of a second sample or distribution.
'''

# scipy.stats 可以画QQ plot 图
n=1000
x=np.array([random.normalvariate(0,1) for _ in range(n)])
y=np.array([random.normalvariate(0,1) for _ in range(n)])
st.probplot(x,plot=plt)

#*****************important********************
nsample=100
ax4 = plt.subplot(224)
x = st.norm.rvs(loc=0, scale=1, size=nsample)
y = st.t.rvs(3, size=nsample)
res = st.probplot(y, plot=ax4)

#4.4 Test of Normality
'''
The null hypothesis is
that the sample comes from a normal distribution and the alternative is that
the sample is from a nonnormal distribution.
'''

'''The Shapiro–Wilk test'''
#test the correlation
nsample=100000
x = st.norm.rvs(loc=0, scale=1, size=nsample)
st.shapiro(x)

#not so accurate for p-value

'''Jarque–Bera test'''
# use the sample skewness and kurtosis coefficients
x=np.random.normal(0,1,100000000)
st.jarque_bera(x)
#only works for large enough samples

'''Anderson–Darling'''
x=np.random.normal(0,1,100000)
st.anderson(x)

'''Kolmogorov–Smirnov tests'''
x=np.random.normal(0,1,1000000)
st.kstest(x,'norm')

'''A small
p-value is interpreted as evidence that the sample is not from a normal
distribution.
'''


#4.5 Boxplots
df=pd.DataFrame(np.random.normal(0,1,1000000))
df.boxplot()


#Data Transformation
''' variance stabilizing
logarithm transformation
'''

#box-cox power transformation:  st.boxcox()
df=pd.Series(np.random.normal(0,1,10000))
df.hist()

#4.7 The geometry of transformations
'''   ????????       '''




''' lab '''


#4.10.2   McDonald's Prices and Returns
data=pd.read_csv('G:\书籍\面试答疑\Python Lab for Statistics and data analysis\datasets\MCD_PriceDaily.csv',index_col='Date')
data.head()
data.index=pd.to_datetime(data.index)
adjPrice=data['Adj Close']

plt.plot(adjPrice)
LogRet=adjPrice.diff()/adjPrice.shift(1)
LogRet.hist(bins=80)
LogRet=LogRet.dropna()
st.probplot(LogRet,plot=plt)



