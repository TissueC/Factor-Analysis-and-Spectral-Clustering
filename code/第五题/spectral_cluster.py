# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN,KMeans
from sklearn.mixture import GaussianMixture
import time


#file='three_cluster.npz'
#data=np.load(file)
#
#x=data['x'][:,np.newaxis]
#y=data['y'][:,np.newaxis]
#X=np.concatenate((x,y),axis=1)
#label=data['label']
#
#
#clustering = SpectralClustering(n_clusters=3,
#                                eigen_solver='arpack',
#                                assign_labels='discretize',
#                                affinity='rbf',
#                                n_neighbors=10,
#                                random_state=0).fit(X)
#
#pred=clustering.labels_
#clustering2 = KMeans(n_clusters=3).fit(X)
#pred2=clustering2.labels_
#
#
#plt.scatter(X[:,0],X[:,1],c=label,s=5)
#plt.savefig('Figures/three_cluster_label.pdf')
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=pred,s=5)
#plt.savefig('Figures/three_cluster_sc.pdf')
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=pred2,s=5)
#plt.savefig('Figures/three_cluster_kmeans.pdf')
#

#file='aniso.npz'
#data=np.load(file)
#x=data['x'][:,np.newaxis]
#y=data['y'][:,np.newaxis]
#X=np.concatenate((x,y),axis=1)
#label=data['label']
#
#clustering = SpectralClustering(n_clusters=3,
#                                eigen_solver='arpack',
#                                assign_labels='discretize',
#                                affinity='nearest_neighbors',
#                                n_neighbors=4,
#                                random_state=0).fit(X)
#pred=clustering.labels_
#clustering2=GaussianMixture(n_components=3)
#clustering2.fit(X)
#pred2=clustering2.predict(X)
#
#plt.scatter(X[:,0],X[:,1],c=label,s=5)
#plt.savefig('Figures/bad_2_label.pdf')
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=pred,s=5)
#plt.savefig('Figures/bad_2_sc.pdf')
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=pred2,s=5)
#plt.savefig('Figures/bad_2_gmm.pdf')


file='X.npy'
X=np.load(file)
clustering=GaussianMixture(n_components=2)
clustering.fit(X)
pred=clustering.predict(X)
plt.scatter(X[:,0],X[:,1],c=pred,s=5)
plt.savefig('Figures/true_scale.pdf')
