# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:39:51 2020

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

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)

r_dt.fit(x,y)

plt.scatter(x,y, color = "red")
plt.plot(x, r_dt.predict(x), color = "blue")

z = x + 0.5
k = x - 0.5 

plt.scatter(x,y, color = "red")
plt.plot(x, r_dt.predict(x), color = "blue")
plt.plot(x, r_dt.predict(z), color = "green")
plt.plot(x, r_dt.predict(k), color = "orange")

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
