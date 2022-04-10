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
import astropy.coordinates as ap_coor
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

#%% 
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pruebas='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
# =============================================================================
# #Choose Arches or Quintuplet
# =============================================================================
choosen_cluster = 'Arches'

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits') if choosen_cluster =='Arches' else Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')

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
# fig, ax = plt.subplots(1,3,figsize=(30,10))
# plotting('ra','dec',ra,dec,0,alpha=0.5)
# plotting('mura','mudec',pmra,pmdec,1,alpha=0.01)
# plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.05)
# # %%
# fig, ax = plt.subplots(1,3,figsize=(30,10))
# plotting_h('ra','dec',ra,dec,0,bins=50,norm=matplotlib.colors.LogNorm())
# plotting_h('mura','mudec',pmra,pmdec,1,bins=50,norm=matplotlib.colors.LogNorm())
# plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.2)

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

samples_dist=5
tree=KDTree(X_stad, leaf_size=2) 
dist, ind = tree.query(X_stad, k=samples_dist) 
# d_KNN=sorted(dist[:,-1])
nn= samples_dist- 1
rev=False
d_KNN=sorted(dist[:,nn],reverse=rev)

kneedle = KneeLocator(np.arange(0,len(X),1), d_KNN, curve='convex', interp_method = "polynomial",direction='increasing' if rev ==False else 'decreasing')
codillo = KneeLocator(np.arange(0,len(X),1), d_KNN, curve='concave', interp_method = "polynomial",direction='increasing' if rev ==False else 'decreasing')
rodilla = round(kneedle.elbow_y, 3)
codo = round(codillo.elbow_y, 3)
# =============================================================================
# # Choose the right epsilon is crucial. I didnt figure it out yet...
# =============================================================================
# epsilon=rodilla
epsilon = codo
# epsilon = min(d_KNN)+0.06


fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(np.arange(0,len(X),1),d_KNN)
ax.set_xlabel('Point') 
ax.set_ylabel('%s-NN distance'%(nn+1))
ax.axhline(rodilla,linestyle='dashed',color='k') 
ax.axhline(codo,linestyle='dashed',color='k') 
ax.axhline(epsilon,linestyle='dashed',color='red') 
ax.text(0,codo, '%s'%(codo))
ax.text(0,rodilla, '%s'%(rodilla))
ax.text(len(X)/2,epsilon, '%s'%(round(epsilon,3)),color='red')
ax.fill_between(np.arange(0,len(X)), codo, rodilla, alpha=0.5, color='grey')


ax.legend(['knee=%s,min=%s, eps=%s, Dim.=%s'%(rodilla,round(min(d_KNN),2),round(epsilon,2),len(X[0]))])



clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)

l_c=clustering.labels_
loop=0
while len(set(l_c)) > 2:#Choose how many clusters you want to find (e.g 2 mean one cluster, 3 means two cluters, etc (l_c are the labes of each cluster plus one for the noise))
    loop +=1
    samples_dist+=1
    clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)
    l_c=clustering.labels_
    print('loop %s min size = %s, cluster found = %s '%(loop,samples_dist, len(set(l_c))-1))
    
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
ax[0].set_title('n of cluster = %s,eps=%s,min size=%s'%(n_clusters,round(epsilon,2),samples_dist))
ax[1].set_title('%s'%(choosen_cluster))
ax[0].invert_xaxis()

for i in range(len(set(l_c))-1):
    plotting('mul','mub',pml[colores_index[i]], pmb[colores_index[i]],0, color=colors[i],zorder=3)
    plotting('l','b',l[colores_index[i]], b[colores_index[i]],1, color=colors[i],zorder=3)
    print(len(pml[colores_index[i]]))
plotting('mul','mub',pml[colores_index[-1]], pmb[colores_index[-1]],0, color=colors[-1],zorder=1)
# plotting_h('mul','mub',X[:,0][colores_index[-1]], X[:,1][colores_index[-1]],0, color=colors[-1],zorder=1)
plotting('l','b',l[colores_index[-1]], b[colores_index[-1]],1, color=colors[-1],zorder=1)


# %%

# %%
# Now that we can find a cluster, we are going to tryint again changing the distance, e.g. zooming in the data
# so, we choose randomnly a cluster point and performn the clustering only on the points within a certain distance
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
clus_gal=arc_gal[colores_index[0]]
pm_clus=pm_gal[colores_index[0]]
# =============================================================================
# # NOte to myself: pm_clus is a Skycoord pm obeject
# # , that is not the same than a Skycoor coord objet. 
# # The former stores coord and pm and, aparently to acces 
# # the proper motions coordinate you have to do it separetly
# # , i.e. pm_clus.pm_l_cob or pm_clu.b(doing pm_clus.pm does not worñ)
# =============================================================================
# pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic

fig, ax = plt.subplots(1,2,figsize=(20,10))



rand = np.random.choice(np.arange(0,len(clus_gal)),1)

rand_clus = clus_gal[rand]
rand_pm = pm_clus[rand]
radio=35*u.arcsec

# idxc, group, d2d,d3d = clus_gal.search_around_sky(rand_clus, radio)
id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(rand_clus,arc_gal, radio)


ax[1].set_title('Radio = %s'%(radio))
ax[0].set_title('%s'%(choosen_cluster))
plotting('l','b',arc_gal.l, arc_gal.b,1)
plotting('l','b',clus_gal.l, clus_gal.b,1)
# plotting('l','b',clus_gal.l[group], clus_gal.b[group],1)
plotting('l','b',arc_gal.l[id_arc], arc_gal.b[id_arc],1,alpha=0.1)

plotting('mul','mub',pm_gal.pm_l_cosb, pm_gal.pm_b,0)
plotting('mul','mub',pm_clus.pm_l_cosb, pm_clus.pm_b,0)
# plotting('mul','mub',pm_clus.pm_l_cosb[group], pm_clus.pm_b[group],0)
plotting('mul','mub',pml[id_arc], pmb[id_arc],0,alpha=0.1)



ax[0].invert_xaxis()
# %%
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
area_l,area_b = arc_gal.l[id_arc],arc_gal.b[id_arc]
area_pml,area_pmb = pml[id_arc], pmb[id_arc]

X_area=np.array([area_pml,area_pmb,area_l,area_b]).T
# X=np.array([area_pml,area_pmb]).T
# X=np.array([area_l,area_b]).T

X_stad_area = StandardScaler().fit_transform(X_area)




samples_dist_area = 5
tree=KDTree(X_stad_area, leaf_size=2) 
dist_area, ind_area = tree.query(X_stad_area, k=samples_dist_area) 
# d_KNN=sorted(dist[:,-1])
nn_area= samples_dist_area- 1
rev=False
d_KNN_area=sorted(dist_area[:,nn],reverse=rev)

kneedle_area = KneeLocator(np.arange(0,len(X_area),1), d_KNN_area, curve='convex', interp_method = "polynomial",direction='increasing' if rev ==False else 'decreasing')
codillo_area = KneeLocator(np.arange(0,len(X_area),1), d_KNN_area, curve='concave', interp_method = "polynomial",direction='increasing' if rev ==False else 'decreasing')
rodilla_area = round(kneedle_area.elbow_y, 3)
codo_area = round(codillo_area.elbow_y, 3)
# =============================================================================
# # Choose the right epsilon is crucial. I didnt figure it out yet...
# =============================================================================
# epsilon_area=rodilla_area
epsilon_area = codo_area
# epsilon_area = min(d_KNN_area)+0.06


fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(np.arange(0,len(X_area),1),d_KNN_area)
ax.set_xlabel('Point') 
ax.set_ylabel('%s-NN distance'%(nn+1))
ax.axhline(rodilla_area,linestyle='dashed',color='k') 
ax.axhline(codo_area,linestyle='dashed',color='k') 
ax.axhline(epsilon_area,linestyle='dashed',color='red') 
ax.text(0,codo_area, '%s'%(codo_area))
ax.text(0,rodilla_area, '%s'%(rodilla_area))
ax.text(len(X_area)/2,epsilon_area, '%s'%(round(epsilon_area,3)),color='red')
ax.fill_between(np.arange(0,len(X_area)), codo_area, rodilla_area, alpha=0.5, color='grey')

clustering_area = DBSCAN(eps=epsilon_area, min_samples=samples_dist_area).fit(X_stad_area)

l_area=clustering_area.labels_

loop_area=0
while len(set(l_area)) > 2:#Choose how many clusters you want to find (e.g 2 mean one cluster, 3 means two cluters, etc (l_c are the labes of each cluster plus one for the noise))
    loop_area +=1
    samples_dist_area+=1
    clustering_area = DBSCAN(eps=epsilon_area, min_samples=samples_dist_area).fit(X_stad_area)
    l_area=clustering_area.labels_
    print('loop %s min size = %s, cluster found = %s '%(loop_area,samples_dist_area, len(set(l_area))-1))
    

n_clusters_area = len(set(l_area)) - (1 if -1 in l_area else 0)
n_noise_area=list(l_area).count(-1)

u_labels_area = set(l_area)
colors_area=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l_area)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


for k in range(len(colors_area)): #give noise color black with opacity 0.1
    if list(u_labels_area)[k] == -1:
        colors_area[k]=[0,0,0,0.1]
        
colores_index_area=[]      
for c in u_labels_area:
    cl_color_area=np.where(l_area==c)
    colores_index_area.append(cl_color_area)


fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title('n of cluster = %s,eps=%s,min size=%s'%(n_clusters_area,round(epsilon_area,2),samples_dist_area))
ax[1].set_title('%s'%(choosen_cluster))
ax[0].invert_xaxis()
# %

for i in range(len(set(l_area))-1):
    plotting('mul','mub',area_pml[colores_index_area[i]], area_pmb[colores_index_area[i]],0, color=colors_area[i],zorder=3)
    plotting('l','b',area_l[colores_index_area[i]], area_b[colores_index_area[i]],1, color=colors_area[i],zorder=3)
    print(len(pml[colores_index_area[i]]))
plotting('mul','mub',pml[colores_index[-1]], pmb[colores_index[-1]],0, color=colors[-1],zorder=1)
# plotting_h('mul','mub',X[:,0][colores_index[-1]], X[:,1][colores_index[-1]],0, color=colors[-1],zorder=1)
plotting('l','b',l[colores_index[-1]], b[colores_index[-1]],1, color=colors[-1],zorder=1)










