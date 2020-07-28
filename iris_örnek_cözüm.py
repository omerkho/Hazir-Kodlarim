# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 00:07:28 2020

@author: Omer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

## Veri Yükleme

data = pd.read_excel("iris.xls")

x = data.iloc[:,:4].values
y = data.iloc[:,4:].values

## Test-Train Bölme
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

## Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state= 0)
logr.fit(X_train,y_train)
logr_pred = logr.predict(X_test)
cm1 = confusion_matrix(y_test,logr_pred)
print(cm1)
print(accuracy_score(y_test,logr_pred))

## KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
cm2 = confusion_matrix(y_test,knn_pred)
print(cm2)
print(accuracy_score(y_test,knn_pred))

## Support Vector Classifier 
from sklearn.svm import SVC
svc = SVC(kernel="poly")
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
cm3 = confusion_matrix(y_test,svc_pred)
print(cm3)
print(accuracy_score(y_test,svc_pred))

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_pred = nb.predict(X_test)
cm4 = confusion_matrix(y_test,nb_pred)
print(cm4)
print(accuracy_score(y_test,nb_pred))

## Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion= "entropy")
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
cm5 = confusion_matrix(y_test,dtc_pred)
print(cm5)
print(accuracy_score(y_test,dtc_pred))

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15, criterion="entropy")
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
cm6 = confusion_matrix(y_test,rfc_pred)
print(cm6)
print(accuracy_score(y_test,rfc_pred))

## ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
