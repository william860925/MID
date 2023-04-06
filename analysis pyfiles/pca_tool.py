import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import geomstats.backend as gs
from sklearn.decomposition import PCA
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from geomstats.geometry.functions import HilbertSphere

class pca:
    '''
        df is your dataframe
        n is number of components to keep      
    '''
    def __init__(self,df,n):
        self.data=df
        self.n=n

    @property
    def tangent(self):
        '''
        This is tangent pca

        HilberSphere() will take dimention as input  
        '''
        manifold = HilbertSphere(self.data.columns.astype(int))
        sinf_data = gs.array([manifold.projection(si) for si in self.data.values]).squeeze()
        mean = FrechetMean(metric=manifold.metric, 
                    method="batch", 
                    max_iter=32, 
                    verbose=False)
        mean.fit(sinf_data)
        mean_estimate = mean.estimate_
        tpca = TangentPCA(metric=manifold.metric, n_components=self.n)
        tpca.fit(sinf_data, base_point=mean_estimate)
        tpca.transform(sinf_data)

        col=['green','darkorange','steelblue']
        for i in range(self.n):
            fig,ax=plt.subplots(1,1,figsize=(5,5))
            ax.plot(self.data.columns.astype(int), mean_estimate.flatten()+tpca.components_[i,:], label='+std', c=col[2])
            ax.plot(self.data.columns.astype(int), mean_estimate.flatten(), label='mean', c=col[1])
            ax.plot(self.data.columns.astype(int), mean_estimate.flatten()-tpca.components_[i,:], label='-std', c=col[0])
            ax.set_title(f'pc{i+1} ({tpca.explained_variance_ratio_[i].round(4)})')
            ax.set_xlabel('wavelength $(nm)$',fontsize='12')
            ax.legend();

    @property
    def normal(self):
        '''
        This is normal pca
        '''

        pca=PCA(n_components=self.n)
        pca.fit_transform(self.data.values)
        evector=pca.components_
        pcs = np.array(evector)

        col=['green','darkorange','steelblue']
        for i in range(self.n):
            fig,ax=plt.subplots(1,1,figsize=(5,5))
            ax.plot(self.data.columns.astype(int), self.data.mean().values+pcs[i], label='+std', c=col[2] )
            ax.plot(self.data.columns.astype(int), self.data.mean().values, label='mean', c=col[1])
            ax.plot(self.data.columns.astype(int), self.data.mean().values-pcs[i], label='-std', c=col[0])
            ax.set_title(f'pc{i+1} ({pca.explained_variance_ratio_[i].round(2)})')
            ax.set_xlabel('wavelength $(nm)$',fontsize='12')
            ax.legend();
              


