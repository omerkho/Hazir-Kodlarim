# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:34:12 2020

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

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=    10, random_state=   0)
rf_reg.fit(x,y)

rf_reg.predict([[6.5]])
rf_reg.predict([[6.6]]) #dectree de 10000 dü

plt.scatter(x,y, color = "red")
plt.plot(x,rf_reg.predict(x), color = "blue")
plt.plot(x, rf_reg.predict(Z), color = "green")
plt.plot(x, rf_reg.predict(K), color = "orange")
plt.show()

#sahada randomforest decision tree'den daha iyidir

from sklearn.metrics import r2_score
print("Random Forest R2 değeri : ")
print(r2_score(y,rf_reg.predict(x)))
print(r2_score(y,rf_reg.predict(Z)))
print(r2_score(y,rf_reg.predict(K)))