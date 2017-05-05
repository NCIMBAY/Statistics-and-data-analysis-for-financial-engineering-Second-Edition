# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:15:27 2017

@author: NCIMBAY
"""
#Regression: Basics

import pandas as pd
import statsmodels.api as sm  
from scipy import stats as st
from sklearn import linear_model
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import math

#9.2

'''linear regression model: least square'''
x=[1,2,3,4,5]
y=[1,2,3,4,5]


'''statsmodels way'''
#this one gives more imformation
a=sm.OLS(x,y)
result=a.fit()
a.predict(1)
result.summary()

'''scipy way'''
st.linregress(x,y)


'''sklearn way'''
X=[[i] for i in x]
clf=linear_model.LinearRegression()
clf.fit(X,y)
Coefficient_x=clf.coef_
clf.predict(X)


#9.3 Multiple Linear Regression
x1=[1,2,3,4,5]
x2=[2,3,4,5,6]
y=[1,2,3,4,5]
x=[x1,x2]
x=np.array(x).T
x=sm.add_constant(x)
a=sm.OLS(y,x)
result=a.fit()
result.summary()


#9.3.1 standard errors, t-Values, and p-Values
'''The tvalue
is the ratio of the estimate to its standard error.

The p-value, associated with testing the
null hypothesis that the coefficient is 0 versus the alternative that it is not
0.

just the linear relationship!
'''    


#9.4 Analysis of Variance, Sum of Squares, and R^2
'''ANOVA Table'''
df=pd.read_csv('G:\\书籍\\面试答疑\\Python Lab for Statistics and data analysis\\datasets\\Garch.csv')
a=sm.OLS('bp ~ cd',data=df).fit()
sm.stats.anova_lm(result)
'''total SS = regression SS + residual error SS.
'''
#something is wrong with the anova in python



#9.4 Degree of Freedom(DF)
'''
The degrees of freedom for regression is p, which is the number of
predictor variables. The total degrees of freedom is n − 1. The residual error
degrees of freedom is n − p − 1.
'''

#9.4.3 Mean Sums of Squares(MS) and F-Tests
''' a little bit hard
'''






#9.4.4  Adjusted R^2
'''
R2 is biased in favor of large models, because R2 is always increased by adding
more predictors to the model, even if they are independent of the response.

adjusted R^2=1-(residual error SS/(n-p-1))/(total SS/(n-1))

adjusted by the freedom
'''

#9.5 Model Selection
''' AIC, BIC, adjusted R^2, and Cp related,
    we have a formula to this
    minimize the AIC or BIC
'''
# signma2 is the mean of residue SS
sigma2=0
p=2
AIC=n*math.log(sigma2)+2*(1+p)
BIC=n*math.log(sigma2)+math.log(n)*(1+p)

#sigmaM is the estimate of sigma2  =sigma2/residue freedom
sigmaM=0
#SSE(p) is the sum of squares for residue error for a model with subset p predictors
SSE(p)=0
Cp=SSE(p)/sigmaM-n+2*(p+1)



#9.6 Collinearity and Variance Inflation
'''Collinearity or multicollinearity: if two predictors
are highly related to each other, it can cause some problems

****variance inflation factor(VIF)
'''
def vif_ridge(corr_x, pen_factors, is_corr=True):
    """variance inflation factor for Ridge regression

    assumes penalization is on standardized variables
    data should not include a constant

    Parameters
    ----------
    corr_x : array_like
        correlation matrix if is_corr=True or original data if is_corr is False.
    pen_factors : iterable
        iterable of Ridge penalization factors
    is_corr : bool
        Boolean to indicate how corr_x is interpreted, see corr_x

    Returns
    -------
    vif : ndarray
        variance inflation factors for parameters in columns and ridge
        penalization factors in rows

    could be optimized for repeated calculations
    """
    corr_x = np.asarray(corr_x)
    if not is_corr:
        corr = np.corrcoef(corr_x, rowvar=0, bias=True)
    else:
        corr = corr_x

    eye = np.eye(corr.shape[1])
    res = []
    for k in pen_factors:
        minv = np.linalg.inv(corr + k * eye)
        vif = minv.dot(corr).dot(minv)
        res.append(np.diag(vif))
    return np.asarray(res)



#Example 9.9 
df=pd.read_csv('G:\\书籍\\面试答疑\\Python Lab for Statistics and data analysis\\datasets\\nelsonplosser.csv',index_col='X.Y.m.d')
#df.index=pd.to_datetime(df.index)
df.index=df.index//10000
gnp_r=df['gnp.r'][df.index>1908]
dif_gnp=gnp_r.diff()
fig=plt.figure(figsize=(16,10))
plt.subplot(2,3,1)
plt.plot(dif_gnp)

dif_log_gnp=pd.Series([math.log(x) for x in gnp_r/gnp_r.shift(1)])
dif_log_gnp.index=gnp_r.index
plt.subplot(2,3,2)
plt.plot(dif_log_gnp)

sqrt_gnp_r=pd.Series([math.log(x) for x in gnp_r/gnp_r.shift(1)])
#the steps of doing that is useful



'''9.7 Partial Residual Plots

The partial residual plot is simply the
plot of the responses against these partial residuals.

'''
#look at the coefficient or the scatter



'''9.8 Centering the Predictors

1.can reduce collinearity in polynomial regression

2.if all predictors are centered, then β0 is the expected value of Y when
each of the predictors is equal to its mean. This gives β0 an interpretable
meaning. In contrast, if the variables are not centered, then β0 is the
expected value of Y when all of the predictors are equal to 0. Frequently,
0 is outside the range of some predictors, making the interpretation of β0
of little real interest unless the variables are centered.






