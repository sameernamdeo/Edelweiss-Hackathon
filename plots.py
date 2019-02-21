# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r'D:\Edelweiss Hackathon\dataset\train\train1.csv')
df = df[df['Symbol']=='und_109']
df = df[df['Date']=='04-01-2015']

x = df.index.values

y = [[]]
y.append(list(pd.Series(df['MidPrice'])))
fig1 = plt.plot(x,y[1],label='MidPrice')

for i in range(1,46):
    y.append(list(pd.Series(df['Feature'+str(i)])))
    plt.plot(x,y[i+1],label='Feature'+str(i))

label = list(pd.Series(df['MidPrice']))
train = df.loc[:,'Feature1':]
sc = StandardScaler()
train = sc.fit_transform(train)

pca = PCA(n_components=1)
train = pca.fit_transform(train)