#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:07:29 2022

@author: amartinez
"""
# %% imports
import astropy.coordinates as ap_coor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pylab as p
from random import seed
from random import random
from sklearn.preprocessing import StandardScaler
# %%Plotting
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
# %%
# Here we are going to have a close up of the group with a putative cluster in it.
# So far these groups are selected by eye, after inspection of the outcome from dbsacan_comparation.py
name='WFC3IR'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))

# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 0.5
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]


gal_coord=SkyCoord(ra = catal[:,5]*u.deg, dec = catal[:,6]*u.deg, frame = 'icrs').galactic
# %%

groups = [1]# groups to be analyzed
radio = 76*u.arcsec # radio(s) of the list you will like to explore
cluster_by = 'all'# or vel or pos
pms=[0,0,0,0]# if you want to substract the value of the movenment of SgrA* give values to pms[pm_ra,pm_dec,pm_l,pm_b]
clus_id = 0# for each gruop up to six cluster could be found. Manually decide which one of them you want to inspec foward


for g in groups:
    
    # ' ra, dec, x, y, pml, pmb, H, Ks'
    cluster = np.loadtxt(pruebas + '%scluster%s_of_group%s.txt'%(pre,clus_id,g))
    ra_ = cluster[:,0]
    dec_ = cluster[:,1]
    print(ra_[0],dec_[0])
    # Process needed for the trasnformation to galactic coordinates
    gal_c = SkyCoord(ra = cluster[:,0]*u.degree, dec = cluster[:,1]*u.degree, frame='icrs').galactic#you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames
    # gal_c=c.galactic
    
    t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))  
    
    index = np.where((catal[:,5] == ra_[0]) & (catal[:,6] == dec_[0]))
    
    id_clus, id_catal, d2d,d3d = ap_coor.search_around_sky(gal_coord[index],gal_coord, radio)

def plotting(namex,namey,x,y,ind,**kwargs):
    
    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s (%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s (%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl
# %
fig, ax = plt.subplots(1,3,figsize=(30,10))
# plotting('l','b',gal_coord[id_catal].l,gal_coord[id_catal].b,1)
# plotting('l','b',gal_c.l,gal_c.b,1)
min_c=min(cluster[:,-2]-cluster[:,-1])
max_c=max(cluster[:,-2]-cluster[:,-1])
min_Ks=min(cluster[:,-1])

plotting('x','y',catal[:,7][id_catal],catal[:,8][id_catal],1)
plotting('x','y',cluster[:,2],cluster[:,3],1)
plotting('mul (mas/yr)','mub (mas/yr)',catal[:,17][id_catal],catal[:,18][id_catal],0)
plotting('mul (mas/yr)','mub (mas/yr)',cluster[:,4],cluster[:,5],0)
plotting('H-Ks','Ks',catal[id_catal][:,3]-catal[id_catal][:,4],catal[id_catal][:,4],2)
plotting('H-Ks','Ks',cluster[:,-2]-cluster[:,-1],cluster[:,-1],2)
ax[2].axvline(min_c,color='r',ls='dashed',alpha=0.5)
ax[2].axvline(max_c,color='r',ls='dashed',alpha=0.5)
ax[2].annotate('%s'%(round(max_c-min_c,3)),(max_c+max_c/5,min_Ks+0.5),color='r')
ax[0].invert_xaxis()
ax[2].invert_yaxis()
# %%
print(cluster[:,2])


