# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 00:53:07 2020

@author: Omer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("maaslar.csv")

#data frame dilimleme
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Numpy dizi (array) dönüsümü
x = x.values
y = y.values

Z = x + 0.5
K = x - 0.4

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=    10, random_state=   0)
rf_reg.fit(x,y)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)

r_dt.fit(x,y)

# Support Vector Regression 
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf', gamma=    0.6)
svr_reg.fit(x_olcekli, y_olcekli)

# Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 2)

x_pr = pol_reg.fit_transform(x)
lr_2 = LinearRegression()
lr_2.fit(x_pr,y)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x,y)

## R2 KIYASLAMA

from sklearn.metrics import r2_score
print("Random Forest R2 değeri : ")
print(r2_score(y,rf_reg.predict(x)))
print(r2_score(y,rf_reg.predict(Z)))
print(r2_score(y,rf_reg.predict(K)))
print("-----------------------------------")
print("Decision Tree R2 değeri : ")
print(r2_score(y,r_dt.predict(x)))
print(r2_score(y,r_dt.predict(Z)))
print(r2_score(y,r_dt.predict(K)))
print("-----------------------------------")
print("Support Vector Regression R2 değeri : ")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
print("-----------------------------------")
print("Polynomial Regression R2 değeri : ")
print(r2_score(y, lr_2.predict(x_pr)))
print("-----------------------------------")
print("Linear Regression R2 değeri : ")
print(r2_score(y,lr.predict(x)))
