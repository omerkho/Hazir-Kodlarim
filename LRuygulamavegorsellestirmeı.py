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
veriler = pd.read_csv('satislar.csv')

'''

# Verilerin Train ve Test icin Bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)

# Verilen Ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

'''
print(veriler)

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]    

satislar2 = veriler.iloc[:,:1].values

print(satislar2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train= sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# linear regression model inşası
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)

# Veri Görsellestirme

plt.plot(x_train,y_train) # (anlamsız cünkü sıra yok)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")










