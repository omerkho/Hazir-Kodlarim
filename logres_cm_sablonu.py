# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 01:03:11 2020

@author: Omer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Veri Yukleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values
print(y)

# Verilerin Train ve Test icin Bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 0)

# Verilen Ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
log_r = LogisticRegression(random_state=0)
log_r.fit(X_train,y_train)

y_pred = log_r.predict(X_test)
print(y_test)
print(y_pred)

# confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# sensitivity = tpos / (tpos+fneg)
# specificity = tneg / (tneg+tpos)
# precision = tpos / (tpos + fpos)
# accuracy = sens*pos/(pos+neg) + spec*neg(pos+neg)
