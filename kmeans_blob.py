# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:20:28 2020

@author: baig
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Making some fictitiout data with make_blobs method

data = make_blobs(n_samples = 500, n_features = 2, centers = 4, 
                  cluster_std = 1.5, random_state = 101)
plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')

# Vary the number of clusters 
kmeans = KMeans(n_clusters = 4)
kmeans.fit(data[0])         # Locations of the instances/samples/data points
kmeans.cluster_centers_		# Predicted cluster centers
kmeans.labels_				# Predicted cluster labels; what the algorithms thinks is True 
                            # (but it's not True, i.e, original)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10, 6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c = kmeans.labels_, cmap = 'rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')
ax1.set_xlabel('X', fontsize = 14)
ax2.set_xlabel('X', fontsize = 14)
ax1.set_ylabel('Y', fontsize = 14)
ax1.set_xlim([-15, 15])
ax2.set_xlim([-15, 15])
ax1.set_ylim([-15, 15])