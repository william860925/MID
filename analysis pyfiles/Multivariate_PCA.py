#import all the libraries
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#import dataset
data = pd.read_excel("Itr_6.xlsx")
Data= data.iloc[:,1:97]

# PCA with Component = 4
pca = PCA(n_couomponents=4)
pca.fit(Data)

#PC1
x = data.iloc[:,0]
y1 = pca.inverse_transform(np.array([-1,0,0,0]))
y2 = Data.values.mean(axis=0)
y3 = pca.inverse_transform(np.array([1,0,0,0]))

plt.plot(x, y1, color='tab:red', lw=2.0) # mean - std 
plt.plot(x, y2, color='k', ls='--', lw=2.0) # mean
plt.plot(x, y3, color='tab:blue', lw=2.0) # mean + std
plt.show()

#PC2
x = data.iloc[:,0]
y1 = pca.inverse_transform(np.array([0,-1,0,0]))
y2 = Data.values.mean(axis=0)
y3 = pca.inverse_transform(np.array([0,1,0,0]))

plt.plot(x, y1, color='tab:red', lw=2.0) # mean - std 
plt.plot(x, y2, color='k', ls='--', lw=2.0) # mean
plt.plot(x, y3, color='tab:blue', lw=2.0) # mean + std
plt.show()

#PC3
x = data.iloc[:,0]
y1 = pca.inverse_transform(np.array([0,0,-1,0]))
y2 = Data.values.mean(axis=0)
y3 = pca.inverse_transform(np.array([0,0,1,0]))

plt.plot(x, y1, color='tab:red', lw=2.0) # mean - std 
plt.plot(x, y2, color='k', ls='--', lw=2.0) # mean
plt.plot(x, y3, color='tab:blue', lw=2.0) # mean + std
plt.show()

#PC4
x = data.iloc[:,0]
y1 = pca.inverse_transform(np.array([0,0,0,-1]))
y2 = Data.values.mean(axis=0)
y3 = pca.inverse_transform(np.array([0,0,0,1]))

plt.plot(x, y1, color='tab:red', lw=2.0) # mean - std 
plt.plot(x, y2, color='k', ls='--', lw=2.0) # mean
plt.plot(x, y3, color='tab:blue', lw=2.0) # mean + std
plt.show()
