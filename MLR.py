#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri On Isleme

#2.1 Veri Yukleme
veriler = pd.read_csv('veriler.csv')
    #pd.read_csv("veriler.csv")

print(veriler)

# Encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
c[:,0] = le.fit_transform(c[:,0])
print(c)
c=ohe.fit_transform(c).toarray()
print(c)


# Numpy dizileri DataFrame DOnusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,0:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

# DataFrame Birlestirme
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

# Verilerin Train ve Test icin Bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)

# Verilen Ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#######ÇÖZÜM BURDAN BAŞLIYOR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


# son halimiz s2'de

boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag], axis = 1)
a_train, a_test, b_train, b_test = train_test_split(veri,boy,test_size = 0.33, random_state = 0)

mlr = LinearRegression()
mlr.fit(a_train,b_train)    
b_pred = mlr.predict(a_test)

# P VALUELERE FALAN BAKMAK İCİN BACKWARD ELIMINATION
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = boy, exog = X_l) # endog bağımlı, exog bağımsız degisken
r = r_ols.fit()
print(r.summary())

# 1- x5 in p değeri çok fazla cıktı
X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary())

# 2 - yeni modelden de x5i de çıkarırsak
X_l = veri.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary())

# 2.aşamada P valueleri düşürdük ama R-Sqaure değeri de düştü.

#test etme
aa_train, aa_test, bb_train, bb_test = train_test_split(X_l,boy,test_size = 0.33, random_state = 0)
mlr.fit(aa_train,bb_train)    
bb_pred = mlr.predict(aa_test)

