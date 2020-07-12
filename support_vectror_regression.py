# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:18:33 2020

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


# linear regression ile

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x,y)


# polynomial regression

from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 2)

x_pr = pol_reg.fit_transform(x)

print(x_pr)

lr_2 = LinearRegression()

lr_2.fit(x_pr,y)

plt.scatter(x,y, color = "black")
plt.plot(x,lr_2.predict(x_pr), color = "red")
plt.show()

# degree 4 olsun, daha iyi sonuc cıktı
pol_reg2 = PolynomialFeatures(degree = 4)
x_pr2 = pol_reg2.fit_transform(x)

print(x_pr2)

lr_3 = LinearRegression()

lr_3.fit(x_pr2,y)

# görsellestirme
plt.scatter(x,y, color = "red")
plt.plot(x,lr.predict(x), color = "black")

plt.scatter(x,y, color = "black")
plt.plot(x,lr_2.predict(x_pr), color = "red")
plt.show()

plt.scatter(x,y, color = "black")
plt.plot(x,lr_3.predict(x_pr2), color = "red")
plt.show()

# tahminler
print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

print(lr_2.predict(pol_reg.fit_transform([[11]])))
print(lr_2.predict(pol_reg.fit_transform([[6.6]])))

print(lr_3.predict(pol_reg2.fit_transform([[11]])))
print(lr_3.predict(pol_reg2.fit_transform([[6.6]])))



# Verilen Ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf', gamma=    0.6)
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color= 'black')

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))




