#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:24:02 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from astropy.coordinates import match_coordinates_sky, SkyOffsetFrame, ICRS,offset_by
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable
from matplotlib import rcParams
import os
import glob
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
# %%
from sklearn.preprocessing import StandardScaler
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
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of

#%% Test on commit
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pruebas='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
#Arches reference point 
center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs')
#Choose Arches or Quintuplet
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits')
# arches = Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')

columnas=str(arches.columns)
arc_coor=SkyCoord(ra=arches['ra*']*u.arcsec+center_arc.ra,dec=arches['dec']*u.arcsec+ center_arc.dec)
# %%
ra, dec =arc_coor.ra, arc_coor.dec
e_ra,e_dec = arches['e_ra*']*u.arcsec, arches['e_dec']*u.arcsec
# %%
pmra, pmdec = arches['pm_ra*']*u.mas/u.yr, arches['pm_dec']*u.mas/u.yr
e_pmra, e_pmdec = arches['e_pm_ra*'].value, arches['e_pm_dec'].value
print(np.std(e_pmra),np.std(e_pmdec))
# %%
m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]

# =============================================================================
# np.savetxt(pruebas + 'arches_for_topcat.txt',np.array([ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value]).T,header='ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value')
# =============================================================================
# %%
arc_gal=arc_coor.galactic
pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic


l,b=arc_gal.l, arc_gal.b
pml,pmb=pm_gal.pm_l_cosb, pm_gal.pm_b
# %% Definition section
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl
# %
def plotting_h(namex,namey,x,y,ind,**kwargs):
    try:
        pl=ax[ind].hexbin(x.value,y.value,**kwargs)
    except:
        pl=ax[ind].hexbin(x,y,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl

# %
def density_plot(a,b,namex, namey, ind, **kwargs):
    
    xy = np.vstack([a,b])
    z = gaussian_kde(xy)(xy)
    pl =ax[ind].scatter(a, b, c=z,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,a.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, b.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl
# %%This is the plottin section
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting('ra','dec',ra,dec,0,alpha=0.5)
plotting('mura','mudec',pmra,pmdec,1,alpha=0.01)
plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.05)
# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting_h('ra','dec',ra,dec,0,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('mura','mudec',pmra,pmdec,1,bins=50,norm=matplotlib.colors.LogNorm())
plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.2)

# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting_h('l','b',l,b,0,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('mul','mub',pml,pmb,1,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('m127-m157','m157',m127-m153,m153,2,norm=matplotlib.colors.LogNorm())

# %%This density plots are cool but takes some time

# =============================================================================
# plt_dim=3
# fig, ax = plt.subplots(1,plt_dim,figsize=(plt_dim*10,10))
# density_plot(l,b,'l','b',0,cmap='inferno')
# density_plot(pml,pmb,'mul','mub',1,cmap='viridis')# add this for log scale in the color map:,norm=matplotlib.colors.LogNorm()
# density_plot( m127-m153,m153,'m127-m153','m157',2,cmap='viridis')
# 
# =============================================================================
# %%

# %%DBSCAN part
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator

    
# %
X=np.array([pml,pmb,l,b]).T
# X=np.array([pml,pmb]).T
# X=np.array([l,b]).T

X_stad = StandardScaler().fit_transform(X)
# X_stad=X

samples_dist=40
tree=KDTree(X, leaf_size=2) 
dist, ind = tree.query(X_stad, k=samples_dist) 
# d_KNN=sorted(dist[:,-1])
nn=1
d_KNN=sorted(dist[:,nn],reverse=True)

kneedle = KneeLocator(np.arange(0,len(X),1), d_KNN, curve='convex', interp_method = "polynomial",direction="decreasing")
rodilla=round(kneedle.elbow_y, 3)
epsilon=rodilla
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(np.arange(0,len(X),1),d_KNN)
ax.set_xlabel('Point') 
ax.set_ylabel('%s-NN distance'%(nn))
ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k') 
ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(rodilla,round(min(d_KNN),2),round(epsilon,2),len(X[0]))])



clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)

l_c=clustering.labels_

n_clusters = len(set(l_c)) - (1 if -1 in l_c else 0)
n_noise=list(l_c).count(-1)

u_labels = set(l_c)
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l_c)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


for k in range(len(colors)): #give noise color black with opacity 0.1
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
        
colores_index=[]      
for c in u_labels:
    cl_color=np.where(l_c==c)
    colores_index.append(cl_color)



fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title('n of cluster = %s,eps=%s,min size=%s'%(n_clusters,epsilon,samples_dist))
ax[0].invert_xaxis()

for i in range(len(set(l_c))-1):
    plotting('mul','mub',pml[colores_index[i]], pmb[colores_index[i]],0, color=colors[i],zorder=3)
    plotting('l','b',l[colores_index[i]], b[colores_index[i]],1, color=colors[i],zorder=3)
    print(len(pml[colores_index[i]]))
plotting('mul','mub',pml[colores_index[-1]], pmb[colores_index[-1]],0, color=colors[-1],zorder=1)
# plotting_h('mul','mub',X[:,0][colores_index[-1]], X[:,1][colores_index[-1]],0, color=colors[-1],zorder=1)
plotting('l','b',l[colores_index[-1]], b[colores_index[-1]],1, color=colors[-1],zorder=1)


# %%
print(X.shape[0])


