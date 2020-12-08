# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:05:14 2020

@author: baig
"""

import numpy as np
import matplotlib.pyplot as plt

''' Define a function for 2D '''

def f(x, y):
    return np.cos(x) + np.sin(y)

x = np.linspace(0, 5, 200)
y = np.linspace(0, 5, 200)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

fig = plt.figure(figsize = (6, 4))
plt.contour(X, Y, Z, 200, cmap = 'RdBu')