# Predicting mid price for one day for a single symbol
# Using PCA and SVR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR

df = pd.read_csv(r'D:\Edelweiss Hackathon\dataset\train\train1.csv')
#dftest = pd.read_csv(r'D:\Edelweiss Hackathon\dataset\test\test1.csv')
df = df[df['Symbol']=='und_109']
df = df[df['Date']=='04-01-2015']
#dftest = dftest[dftest['Symbol']=='und_109']
#dftest = dftest[dftest['Date']=='01-06-2015']
x = df.index.values

label = list(pd.Series(df['MidPrice']))
train = df.loc[:49,'Feature1':]
test = df.loc[50:,'Feature1':]

sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.fit_transform(test)

pca = PCA(n_components=2)
train = pca.fit_transform(train)
test = pca.fit_transform(test)

price = pd.Series(df['MidPrice'])
trainprice = np.array(price[:50])
testprice = np.array(price[50:])

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(train,trainprice).predict(test)
y_lin = svr_lin.fit(train,trainprice).predict(test)
y_poly = svr_poly.fit(train,trainprice).predict(test)


#For 1 feature pca, svr mse = 21.09,14.67,19.98
#For 2 feature pca, svr mse = 44.05,12.8,21.9
#For 3 feature pca, svr mse = 19.41,26.55,21.72
#For 4 feature pca, svr mse = 20.82,27.58,28.46
mse_rbf,mse_lin,mse_poly = 0,0,0
for i in range(23):
    mse_rbf += (label[50+i]-y_rbf[i])**2
    mse_lin += (label[50+i]-y_lin[i])**2
    mse_poly += (label[50+i]-y_poly[i])**2

mse_rbf,mse_lin,mse_poly = mse_rbf/23,mse_lin/23,mse_poly/23

lw = 2
plt.plot(x[:50],trainprice, color='darkorange', label='data')
plt.plot(x[50:], y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(x[50:], y_lin, color='red', lw=lw, label='Poly')
plt.plot(x[50:], y_poly, color='c', lw=lw, label='Linear')
plt.plot(x[50:],testprice,color='darkorange',label='test')
