# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:29:41 2019

@author: almuhib
"""


# This python code calculates the directional elasticity of tobermorite-14 angstrom
# (a triclinic crystal with 21 elastic constants) in 3D Eucledean coordinates
# from the 6-by-6 compliance tensor

# This code is general enough to calculate the directional elasticity all 7 crystal systems

# The coordinate transformation equation is, 

# S_ijkl = a_im * a_jn * a_ko * a_lp * S_mnop
# Here, S_ijkl = Compliance tensor in the new coordinate
# S_mnop = Compliance tensor in the old coordinate
# a_im, a_jn, a_ko, a_lp = direction cosines



'''---------------Declare some modules ---------------'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib import rc
from matplotlib import ticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict
import matplotlib.colors as colors
from matplotlib.colors import Normalize
#from matplotlib.mlab import bivariate_normal

''' ---------- DECLARE FIGURE SIZE-----------'''

figW = 3.5; 
GR = 1.618; 			# figW = figure width, GR = golden ratio
figH = figW * GR;       # figure height


''' ---------- DECLARE FONT ATTRIBUTES -----------'''

rc('font', **{'family':'serif','serif':['Times New Roman']})		
rc('text', usetex=True)
font = {'family' : 'normal',    # Serif family fonts usually looks better 
        'weight' : 'bold',
        'size'   : 12}
mpl.rc('font', **font)


'''------------ DECLARE TENSOR ELEMENTS------------'''

mat = (6,6)
Cij = np.genfromtxt('t14allZZ.const', dtype=float)

Cij0=Cij[0:6,0:6]      # T14
#Cij0=Cij[0:6,6:12]      # T14/GS
#Cij0 = Cij[0:6,12:18]      # T14/GS25
#Cij0=Cij[0:6,18:24]      # T14/GS50
#Cij0=Cij[0:6,24:30]      # T14/GS75
#Cij0=Cij[0:6,30:36]      # T14/GS100
Sij0 = np.linalg.inv(Cij0)
#Sij0=np.linalg.inv(Cij0)

'''----------- CONVERT SPHREICAL TO CARTESIAN COORDINATES -------'''

N = 100
theta, phi = np.linspace(0, 2 * np.pi, N), np.linspace(0, np.pi, N)
THETA, PHI = np.meshgrid(theta, phi)

'''------ The following equations show how carteisan COORDINATES
            are related to the spherical coordinates---------'''

x = np.cos(THETA) * np.sin(PHI)
y = np.sin(THETA) * np.sin(PHI)
z = np.cos(PHI)
l = np.column_stack((x.reshape(N * N, 1), y.reshape(N * N, 1),
    z.reshape(N * N, 1)))   # l = direction cosine


'''---------- Calculate the elastic modulus in all directions
                selected above using x, y, z, and l (direction cosine) --------'''

norms = np.linalg.norm(l, axis =1)
E0inv_XY= (l[:,0]**4*Sij0[0,0] + l[:,1]**4*Sij0[1,1] + l[:,2]**4*Sij0[2,2] + l[:,1]**2*l[:,2]**2*Sij0[3,3]

    + l[:,0]**2*l[:,2]**2*Sij0[4,4] + l[:,0]**2*l[:,1]**2*Sij0[5,5] + 2*l[:,0]**2*l[:,1]**2*Sij0[0,1]

    + 2*l[:,0]**2*l[:,2]**2*Sij0[0,2] + 2*l[:,1]**2*l[:,2]**2*Sij0[1,2] + 2*l[:,0]**2*l[:,1]*l[:,2]*Sij0[0,3]

    + 2*l[:,0]**3*l[:,2]*Sij0[0,4] + 2*l[:,0]**3*l[:,1]*Sij0[0,5]

    + 2*l[:,1]**3*l[:,2]*Sij0[1,3] + 2*l[:,1]**2*l[:,0]*l[:,2]*Sij0[1,4] + 2*l[:,1]**3*l[:,0]*Sij0[1,5]

    + 2*l[:,2]**3*l[:,1]*Sij0[2,3] + 2*l[:,2]**3*l[:,0]*Sij0[2,4] + 2*l[:,2]**2*l[:,0]*l[:,1]*Sij0[2,5]

    + 2*l[:,2]**2*l[:,1]*l[:,0]*Sij0[3,4] + 2*l[:,1]**2*l[:,2]*l[:,0]*Sij0[3,5]

    + 2*l[:,0]**2*l[:,2]*l[:,1]*Sij0[4,5])

'''
    + l[:,0]**2*l[:,2]**2*Sij2[4,4] + l[:,0]**2*l[:,1]**2*Sij2[5,5] + 2*l[:,0]**2*l[:,1]**2*Sij2[0,1]

    + 2*l[:,0]**2*l[:,2]**2*Sij2[0,2] + 2*l[:,1]**2*l[:,2]**2*Sij2[1,2] + 2*l[:,0]**2*l[:,1]*l[:,2]*Sij2[0,3]

    + 2*l[:,0]**3*l[:,2]*Sij2[0,4] + 2*l[:,0]**3*l[:,1]*Sij2[0,5]

    + 2*l[:,1]**3*l[:,2]*Sij2[1,3] + 2*l[:,1]**2*l[:,0]*l[:,2]*Sij2[1,4] + 2*l[:,1]**3*l[:,0]*Sij2[1,5]

    + 2*l[:,2]**3*l[:,1]*Sij2[2,3] + 2*l[:,2]**3*l[:,0]*Sij2[2,4] + 2*l[:,2]**2*l[:,0]*l[:,1]*Sij2[2,5]

    + 2*l[:,2]**2*l[:,1]*l[:,0]*Sij2[3,4] + 2*l[:,1]**2*l[:,2]*l[:,0]*Sij2[3,5]

    + 2*l[:,0]**2*l[:,2]*l[:,1]*Sij2[4,5])

'''


E0 = 1/E0inv_XY
#E2=1/E2inv_XY
E0 = E0.reshape(N , N)

Xb = E0 * np.cos(THETA) * np.sin(PHI)
Yb = E0 * np.sin(THETA) * np.sin(PHI)
Zb = E0 * np.cos(PHI)


''' ------- PLOTTING 3D VISUALIZATION OF
            ELASTIC ANISOTROPY --------- '''

fig = plt.figure(figsize = (8 , 6)); fig.clf()
ax = fig.add_subplot(111, projection = '3d')


''' ------- DEFINING FUNCTION FOR CALCULATING MIDPOINTS OF 
            COLORBAR------- '''
            
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#norm = mpl.colors.Normalize(vmin=0, vmax=135)
norm = MidpointNormalize(midpoint = 77, vmin = 0, vmax = 154)
m = cm.ScalarMappable(cmap = plt.cm.jet, norm = norm)
m.set_array([])
ticks = np.linspace(0, 154, 6)
cbar = plt.colorbar(m, shrink = 0.75, ticks = ticks, format = '%.1f')
hfont = {'fontname':'footlight MT light'}

plt.setp(cbar.ax.yaxis.get_ticklabels(), weight = 'bold', fontsize=16)
cbar.set_label(r' Elastic Modulus, E (GPa)', labelpad = 25, size=16, rotation=270, **hfont)
cbar.ax.tick_params(labelsize = 16)
surf=ax.plot_surface(Xb, Yb, Zb, rstride = 1, cstride = 1, color = "r", linewidth = 0.25, alpha =1, facecolors=plt.cm.jet(norm(E0)), shade =False)

ax.set_xlabel(r' X', labelpad = 6, fontsize = 16)
ax.set_ylabel(r' Y', labelpad = 6, fontsize = 16)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel(r' Z', labelpad = 9, fontsize = 16, rotation = 90)
plt.xticks(np.linspace(-100,100, 6, endpoint = True), fontsize = 16)
plt.yticks(np.linspace(-100,100, 6, endpoint = True), fontsize = 16)

plt.setp(ax.get_xticklabels(), rotation = 0, rotation_mode = "anchor", horizontalalignment = 'left')
plt.setp(ax.get_zticklabels(), rotation = 0, rotation_mode = "anchor", horizontalalignment = 'center')
plt.setp(ax.get_yticklabels(), rotation = 0, rotation_mode = "anchor", horizontalalignment = 'right')
#ax.get_xaxis().set_ticks([])
#ax.get_yaxis().set_ticks([])
ax.set_xlim3d(-150 , 150)       # setting plot area limits in X direction
ax.set_ylim3d(-150 , 150)       # setting plot area limits in Y direction
ax.set_zlim3d(-150 , 150)       # setting plot area limits in Z direction
zticks = np.linspace(-100 , 100 , 6)
ax.set_zticks(zticks)

ax.tick_params(axis= 'x', which = 'major', pad = -5)
ax.tick_params(axis= 'y', which = 'major', pad = -5)
ax.tick_params(axis= 'z', which = 'major', pad = 10, labelsize=16)
ax.grid(False)

ax.view_init(30 , 45)
fig.tight_layout()
fig.canvas.draw()
#plt.savefig('D:/NICOM6/CBM/image/t1silgrapall_3d.jpg',format='jpg',
 #           bbox_inches = 'tight',dpi=500)
#plt.savefig('C:/Users/almuhib/Research/T14paper-02.12.2019/image/t14grap25ZZ_elastic.jpg',format='jpg', bbox_inches = 'tight',dpi=500)
#plt.savefig('D:/T14paper-02.12.2019/image/t14grap75R_elastic.jpg',format='jpg', bbox_inches = 'tight',dpi=1000)
