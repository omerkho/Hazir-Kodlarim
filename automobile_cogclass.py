# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:29:49 2020

@author: Omer
"""

import pandas as pd
import numpy as np
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header = None)
df.head(15) # ilk 15 row
df.tail(10)  # son 10 row
header = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style",
          "drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type",
          "num-of-cylinders","engine-size","fuel-system","bore","stroke","compres-ratio","horsepower",
          "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = header
path = "C:/Users/Ömer/Desktop/OKUL/MAKİNE ÖĞRENMESİ/DATA ANALYSIS WITH PYTHON/automobile.csv"
df.to_csv(path) #exporting data
df.dtypes #columnların datatype'ını gösterir
df.describe(include ="all") #istatistiksel özet
df.info()

## MODULE 2- DATA WRANGLING
df = df.replace({'?' : np.nan}) #bütün ?'lerini nan yaptık
df.dropna(subset = ["price"], axis = 0, inplace = True) # price'ı NaN, axis=0 row'u siler.
df = df[df.price != "?"] # bütün ?'lerini nan yapmadan price'ı ? değerli row'ları böyle sileriz.
df["normalized-losses"] = df["normalized-losses"].astype(float) #object olan type float yaptık
mean = df["normalized-losses"].mean()
df["normalized-losses"] = df["normalized-losses"].replace(np.nan, mean) #nan'ları mean yaptık

df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg" : "city-L/100km"}, inplace = True)
df["length"] = (df["length"]-df["length"].mean())/df["length"].std() #tool kullanmadan z score standardize etmek

df["price"] = df["price"].astype(float)
bins = np.linspace(min(df["price"]),max(df["price"]),4)
group_names = ["Low","Medium","High"]
df["price-binned"] = pd.cut(df["price"], bins, labels = group_names, include_lowest = True)
#yukarıdaki 4 satırda önce price'ı float yaptık sonra 4'e bölüp low,med,high olarak gruplandırdık.
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.hist(df["price"],bins=3)
pyplot.bar(group_names,df["price-binned"].value_counts())

one_hot = pd.get_dummies(df["fuel-type"])
df = df.drop("fuel-type",axis=1)
one_hot.rename(columns={'diesel':'fuel-diesel', 'gas': 'fuel-gas'}, inplace=True)
df = df.join(one_hot)
df.columns
df = df[['symboling', 'normalized-losses', 'make', 'fuel-diesel', 'fuel-gas','aspiration', 'num-of-doors',
       'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length',
       'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
       'engine-size', 'fuel-system', 'bore', 'stroke', 'compres-ratio',
       'horsepower', 'peak-rpm', 'city-L/100km', 'highway-mpg', 'price',
       'price-binned']] #tekrar sıraladık column'ları

df.to_csv("wrangling_automobile.csv")


## MODULE 3 - EXPLORATORY DATA ANALYSIS
df.describe()
drive_wheel_counts = df["drive-wheels"].value_counts()
import seaborn as sns
sns.boxplot(x = "drive-wheels", y="price", data = df)
x = df["engine-size"]
y = df["price"]
plt.scatter(x,y)
plt.title("Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()

df_test = df[["drive-wheels","body-style","price"]]
df_grp = df_test.groupby(["drive-wheels","body-style"], as_index = False).mean()
df_grp.sort_values(by="price",ascending = False) #büyükten küçüğe sıraladık price'a göre
df_pivot = df_grp.pivot(index="drive-wheels",columns="body-style")
df_pivot.fillna(0) # kategorilere göre özelliklerin kesişimindeki değeri veriyor
plt.pcolor(df_pivot ,cmap = "RdBu")
plt.colorbar()
plt.show()

#anova
df_anova = df[["make","price"]]
grouped_anova = df_anova.groupby(["make"])
from scipy import stats
anova1 = stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("subaru")["price"])
anova1 # f score düşük ve p value çok fazla, keskin ayrımlar yok.
anova2 = stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("jaguar")["price"])
anova2 # f score fazla ve p value çok düşük, keskin ayrımlar var kıyas yapılabilir

#corelation
sns.regplot(x="engine-size",y="price", data=df)
plt.ylim(0,) # positif bir korelasyon var ikisi arasında
sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,) # negatif korelasyon
sns.regplot(x="peak-rpm",y="price",data=df)
plt.ylim(0,) # çok weak bir korelasyon var, price buna bağlı değil
# korelasyon 1 ve -1'e yaklastıkca artar 0'a yaklastıkca azalır
df.horsepower = df.horsepower.astype(float)
df.horsepower = df.horsepower.replace(np.nan, mean) #nan'ları mean yaptık istatistik için?
pearson_coef, p_value = stats.pearsonr(df["horsepower"],df["price"])
pearson_coef #pearson korelasyonunun fazla olması iyi
p_value  # p value'de düşük


## MODULE 4 - MODEL DEVELOPMENT
sns.residplot(df["highway-mpg"],df["price"])
ax1 = sns.distplot(df["price"], hist = False, color = "r", label = "Actual Value")
# sns.distplot(Yhat,hist=False,color="b",label="Fitted Values",ax=ax1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
y = df["price"]
x = df[["horsepower","curb-weight"]]
pr = PolynomialFeatures(degree=2, include_bias= False)
x_polly = pr.fit_transform(df[["horsepower","curb-weight"]])
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(x,y) #pipe birçok işlemi bir arada yapmaya yarar


## MODULE 5 - MODEL EVALUATION
# cross_val_score
# cross_val_pred
# model çok basit olursa data için mesela poly olana lr yaparsan : underfitting
# model çok flexible olursa data için oturmazsa : overfitting 
# aşağıda farklı degreeler için polynomialregression'ın hataları verilmis örnek kod var
Rsqu_test = []
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree = n)
    x_train_pr = pr.fit_transform(x_train[["horsepower"]])
    x_test_pr = pr.fit_transform(x_test[["horsepower"]])
    lr.fit(x_train_pr,y_train)
    Rsqu_test.append(lr.score(x_test_pr,y_test))

#ridge regression
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha= 0.1)
RidgeModel.fit(x,y)    
Yhat = RidgeModel.predict(x)

#grid search
from sklearn.model_selection import GridSearchCV
parameters1 = [{"alpha" : [0.001,0.1,1,10,100,1000,10000,100000,1000000]}]
rr = Ridge()
Grid1 = GridSearchCV(rr, parameters1, cv=4)
Grid1.fit(df[["horsepower","curb-weight","engine-size","highway-mpg"]],y)
Grid1.best_estimator_
scores= Grid1.cv_results_
scores["mean_test_score"]
parameters2 = [{"alpha": [1,10,100,1000], "normalize" : [True,False]}]
Grid2 = GridSearchCV(rr, parameters2, cv=4)
Grid2.fit(df[["horsepower","curb-weight","engine-size","highway-mpg"]],y)
Grid2.best_estimator_
scores= Grid2.cv_results_
for param,mean_val,mean_test in zip(scores["params"],scores["mean_test_score"],scores["mean_train_score"]):
    print(param, "R^2 on test data:",mean_val, "R^2 on train data:",mean_test)