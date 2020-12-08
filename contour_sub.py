# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:33:54 2020

@author: baig
"""

########################################
##### Plotting Contour Sub Plots #######
########################################

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
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (figW, figH), sharex = True, sharey = True)
ax[0].contour(X, Y, Z, 20, cmap = 'RdBu')
filled = ax[1].contourf(X, Y, Z, 20, cmap = 'RdBu')
fig.colorbar(filled, ax = ax, ticks = np.linspace(np.min(Z), np.max(Z), 5),)
ax[0].set_xlabel('x (unit)', fontsize = 14)
ax[0].set_ylabel('y (unit)', fontsize = 14)
ax[0].set_xticks(np.linspace(0, 5, 6))
ax[0].set_yticks(np.linspace(0, 5, 6))
ax[0].tick_params(axis="x", labelsize=12) 
ax[1].tick_params(axis="x", labelsize=12) 
ax[0].set_title('unfilled contour', fontsize = 10)
ax[1].set_title('filled contour', fontsize = 10)