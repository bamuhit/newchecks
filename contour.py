# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:05:14 2020

@author: baig
"""

###################################
##### Plotting Contour Plot #######
###################################

''' Importing modules '''
import numpy as np
import matplotlib.pyplot as plt

''' Define a function Sin and Cosine on 2D '''
def f(x, y):
    return np.cos(x) ** 2 + np.sin(y) ** 2


x = np.linspace(0, 5, 200)
y = np.linspace(0, 5, 200)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

''' Plotting contour '''
figW = 3.5
GR = 1.609
figH = 3.5/GR
fig = plt.figure(figsize = (figW, figH))
plt.contour(X, Y, Z, 200, cmap = 'RdBu')
plt.colorbar(extend = 'both', ticks = np.linspace(np.min(Z), np.max(Z), 5))
plt.xlabel('x (unit)', fontsize = 14)
plt.ylabel('y (unit)', fontsize = 14)
plt.xticks(np.linspace(0, 5, 6), fontsize = 12)
plt.yticks(np.linspace(0, 5, 6), fontsize = 12)
fig.tight_layout()