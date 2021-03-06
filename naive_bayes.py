# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 01:46:45 2020

@author: Omer
"""

# tahmin edeceğin kolon continuous ise gausian
# tahmin edeceğin kolon nominal ise multinomial
# tahmin edeceğin kolon binary ise bernoulli

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

# Verilerin Train ve Test icin Bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 0)

# Verilen Ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print(score)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# accuracy score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print(score)
