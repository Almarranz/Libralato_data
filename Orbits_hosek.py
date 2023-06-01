#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:38:10 2023

@author: amartinez
"""

# Proyects Hosek orbits on the NSD and compares with the proper motions for
# the massives stars


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from astropy.io import ascii
import astropy.coordinates as ap_coor
from astropy.io import fits
# %%plotting pa    metres
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 15})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# 
# %%
orbits='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/Orbits/'
morralla ='/Users/amartinez/Desktop/morralla/'
cls_to_throw = '/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/'
pruebas = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
# Arches = ascii.read(orbits + 'orbits_arches_pro.fits') 
choosen = 'Quintuplet'
if choosen == 'Arches':
    cluster = fits.open(orbits + 'orbits_arches_pro.fits',memmap=True) 
elif choosen == 'Quintuplet':
    cluster = fits.open(orbits + 'orbits_quint_pro.fits',memmap=True) 
 
# %%
# File details
# In the tables, each row represents a possible orbit for the cluster, each with a probability assigned. 
# Taken together, these orbits define the probability distribution of possible orbits for the clusters. 
# Each orbit is integrated in 200 time-steps from 0 - 10 Myr.

# Note that some columns are 2D columns (e.g., Nrows x Nelements per row). 
# Astropy tables can handle this, but I'm not sure pandas can. 

# Columns:
# weights: The probability of the orbit (sums to 1 over entire file)
# logLike: The log-likelihood value in the model for the orbit, from Hosek+22

# galpy_x0, galpy_y0, galpy_z0: The birth position of the orbit, in galpy coordinates (pc relative to SgrA*)
# galpy_vx0, galpy_y0, galpy_z0: The birth velocity of the orbit, in galpy coordinates (km/s relative to SgrA*)

# galpy_x, galpy_y, galpy_z: The position of the cluster on the orbit as a function of time, in galpy coordinates (pc relative to SgrA*)
# galpy_vx, galpy_vy, galpy_vz: The velocity of the cluster on the orbit as a function of time, in galpy coordinates (km/s relative to SgrA*)

# radius: The radius of the cluster as a function of time (pc relative to SgrA*)
# t_orb: Time of the orbit step (in Myr)

# obs_l, obs_b: The galactic longitude (l) and latitude (b) of the cluster on the orbit as a function of time (degrees)
# obs_pml_edr3, obs_pmb_edr3: The proper motion in galactic longitude and latitude of the cluster as a function of time (mas/yr, in Gaia EDR3 coords)

# sgra_phase: a measure of orbital phase relative to SgrA*. I didn't end up really using this, so you can ignore

# period: The period of the orbit (Myr)
# apo: The apoapse of the orbit (pc)
# peri: the periapse of the orbit (pc)
# dlos: The present-day line-of-sight distance of the cluster from SgrA* predicted by the orbit (pc relative to SgrA*)

# clust_logAge: The current age of the cluster in the particular orbit model, in log(years)


ind = -1
t1, t2=0, 80
ti = cluster[1].data['t_orb'][ind][t1]
tf = cluster[1].data['t_orb'][ind][t2]
l_all = cluster[1].data['obs_l']
b_all = cluster[1].data['obs_b']
z_all = cluster[1].data['galpy_z']

z = cluster[1].data['galpy_x'][ind][t1:t2]
x = cluster[1].data['galpy_y'][ind][t1:t2]
y = cluster[1].data['galpy_z'][ind][t1:t2]

z0 = cluster[1].data['galpy_x0'][ind]
x0 = cluster[1].data['galpy_y0'][ind]
y0 = cluster[1].data['galpy_z0'][ind]

z_all = cluster[1].data['galpy_x'][ind]
x_all = cluster[1].data['galpy_y'][ind]
y_all = cluster[1].data['galpy_z'][ind]


l_pc = np.radians(cluster[1].data['obs_l'][ind][t1:t2])*8.2e3
b_pc = np.radians(cluster[1].data['obs_b'][ind][t1:t2])*8.2e3
fig, ax = plt.subplots(1,2)
ax[0].set_title('Orbit # %s (%.1f to %.1f Myr)'%(ind, ti, tf))
# ax[1].scatter(l_all[ind],b_all[ind], color = 'k', alpha = 0.0)
z_ = ax[1].scatter(x,y, c = -z,cmap = 'viridis_r')
clb = plt.colorbar(z_, label = 'Z [pc]', location='bottom',)
# clb.invert_yaxis()
ax[1].invert_xaxis()

ax[1].set_title('%s cluster'%(choosen))
# ax[0].scatter(l_all[ind],b_all[ind], color = 'k', alpha = 0.0)
ax[0].scatter(l_pc,-z)
# plt.colorbar(y, label = 'Z [pc]', location='bottom')
ax[0].invert_xaxis()

# fig = plt.figure(figsize =(8,8))
# ax = fig.add_subplot(projection='3d')

# ax.set_xlabel('x')
# ax.set_ylabel('z')
# ax.set_zlabel('y')
# ax.xaxis.labelpad = 20
# ax.yaxis.labelpad = 20
# # ax.scatter(0,0,0, color = 'k', s = 200)
# ax.invert_xaxis()
# # ax.invert_zaxis()
# # ax.scatter(x_all,-z_all,y_all, color = 'black', alpha = 0.1)
# ax.scatter(x,-z,y, c = -z, cmap = 'viridis_r')
# ax.scatter(x0,-z0,y0, color = 'black', s =200)




l_deg = cluster[1].data['obs_l'][ind][t1:t2]
b_deg = cluster[1].data['obs_b'][ind][t1:t2]

# fig, ax = plt.subplots(1,1)
# ax.scatter(l_deg,b_deg)

with open(pruebas + 'Orbit_%s_%.1f.reg'%(choosen, tf),'w') as f:
    reg_color = 'green' if choosen == 'Arches' else 'blue'
    f.write('\n'.join(('# Region file format: DS9 version 4.1',
                      'global color=%s dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'%(reg_color),
                      'galactic\n')))
    f.close()
    
for cc in range(len(l_deg)-1):
    with open(pruebas + 'Orbit_%s_%.1f.reg'%(choosen, tf),'a') as f:
        if z[cc] > 0:
            # f.write('circle(%s,%s,20")# fill = 1 color = blue\n'%(l_deg[cc],b_deg[cc]*10))
            f.write('line(%s,%s,%s,%s)# line  = 0 0 width = 4\n'%(l_deg[cc],b_deg[cc],l_deg[cc+1],b_deg[cc+1]))

        elif z[cc] < 0:
            # f.write('circle(%s,%s,20") \n '%(l_deg[cc],b_deg[cc]*10))
            f.write('line(%s,%s,%s,%s)# line  = 0 0 dash = 1 width = 2\n'%(l_deg[cc],b_deg[cc],l_deg[cc+1],b_deg[cc+1]))







