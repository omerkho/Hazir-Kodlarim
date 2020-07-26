# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:22:40 2020

@author: Omer
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

df = pd.read_csv("hokey.csv")
df['Date'] <- pd.to_datetime(df['Date'], format = "%Y-%m-%d")
df["goal_dif"] = df['G.1'] - df['G']

df['home_win'] = np.where(df['goal_dif'] > 0,1,0)
df['home_loss'] = np.where(df['goal_dif'] < 0,1,0)

df_visitor = pd.get_dummies(df['Visitor'], dtype = np.int64)
df_home = pd.get_dummies(df['Home'], dtype = np.int64)
df_model = df_home.sub(df_visitor)
df_model['goal_dif'] = df['goal_dif']

df_train = df_model

lr = Ridge(alpha = 0.001)
x = df_train.drop(['goal_dif'], axis = 1)
y = df_train['goal_dif']
y = np.int64(y)
lr.fit(x,y)

df_ratings = pd.DataFrame(data = {'team' : x.columns, 'rating' : lr.coef_})
df_ratings
df_ratings.sort_values(by = 'rating', ascending = False)
