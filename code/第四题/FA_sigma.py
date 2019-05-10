# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:57:12 2019

@author: Administrator
"""


import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
#Generate Data

#for m in range(2,11):
for noise_level in [0,0.03,0.06,0.09,0.12,0.15,
                    0.18,0.21,0.24,0.27,0.3]:
    aic_right,bic_right=0,0
    for _ in range(100):
        n_sample=np.random.randint(20,1001)
        m=np.random.randint(2,11)
        n=np.random.randint(m+1,21)
        #noise_level=0.3*np.random.random()
        mu=np.random.random()
        A=np.random.random((m,n))
        
    #        print('configs ---------\n \
    #              n_sample: %d \n \
    #              m:%d \n \
    #              n:%d \n \
    #              noise level: %f \n \
    #              mu: %f \n-----------------'%(n_sample,m,n,noise_level,mu))
        y=np.random.normal(0,1,(n_sample,m))
        noise=np.random.normal(0,noise_level,(n_sample,n))
        X=np.dot(y,A)+mu+noise
        
        
        plot_x=list()
        aic_list=list()
        bic_list=list()
        for n_components in range(2,n):
            n_params= n*n_components+n-n_components*(n_components-1)/2
            
            transformer = FactorAnalysis(n_components=n_components, random_state=0)
            X_transformed=transformer.fit_transform(X)
            plot_x.append(n_components)
            
            aic=transformer.score(X)*n_sample - n_params
            bic=transformer.score(X)*n_sample - 0.5*n_params*np.log(n_sample)
            aic_list.append(aic)
            bic_list.append(bic)
            #print('n_components:',n_components,'\n aic: ',aic,'\t bic: ',bic)
        if np.argmax(aic_list)+2==m:
            aic_right+=1
        if np.argmax(bic_list)+2==m:
            bic_right+=1
    print('noise level:',noise_level)
    print('aic:',aic_right/100)
    print('bic:',bic_right/100)
#        print('aic-argmax: ',np.argmax(aic_list)+2)
#        print('bic-argmax: ',np.argmax(bic_list)+2)
#plt.plot(np.array(plot_x),np.array(aic_list),'o-')
#plt.plot(np.array(plot_x),np.array(bic_list),'o-')
#plt.xticks(plot_x)
#plt.show()

