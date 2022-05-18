# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:04:07 2022

@author: Daniel
"""

#Section 4: Silhouette metric to evaluate model performance
#Section 4.1: Set cluster=3
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt # gráficos

X,y=datasets.make_blobs(n_samples=150,
                        n_features=2,
                        centers=3,
                        cluster_std=0.5,
                        shuffle=True,
                        random_state=0)
km=KMeans(n_clusters=3,
          init='k-means++',
          n_init=10,
          max_iter=300,
          tol=1e-4,
          random_state=0)
y_km=km.fit_predict(X)

cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
silhouette_score_cluster_3=silhouette_score(X,km.labels_)
print("Silhouette Score When Cluster Number Set to 3: %.3f" % silhouette_score_cluster_3)
silhouette_vals=silhouette_samples(X,y_km,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]
plt.figure(8)
for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.0)
    y_ax_lower+=len(c_silhouette_vals)

silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.yticks(yticks,cluster_labels+1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette Coefficients")
plt.show()