#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:28:52 2022

@author: amartinez
"""

# =============================================================================
# Here we are going to divide section A in smalles LxL areas, using an especific
# epsilon for that area.
# =============================================================================
# %%imports
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
import glob
from sklearn.preprocessing import StandardScaler
import os
import math
# %%plotting parametres
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
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

section = 'D'#selecting the whole thing
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%(section)

# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%
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
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
# catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))
# %%
# clus_test = np.loadtxt(pruebas + 'dbs_cluster1_of_group89.txt')
m1 = -0.86
m = 1
step = 3300

color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)

for f_remove in glob.glob(pruebas + 'subsec_%s/subsec*'%(section)):
    os.remove(f_remove)

missing =0
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8])
fila =-1
lim_pos_up, lim_pos_down = 1000, -18300 #intersection of the positives slopes lines with y axis,
lim_neg_up, lim_neg_down =24000,17300 #intersection of the negayives slopes lines with y axis,

yg_up =  lim_pos_up + m*catal[:,7]
yg_down =  lim_pos_down + m*catal[:,7]

# distancia entre yg_up e yg_down
dist_pos = abs((-m*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))

yr_up = lim_neg_up + m1*catal[:,7]
yr_down = lim_neg_down + m1*catal[:,7]
# distancia entre yg_up e yg_down
dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
ang = math.degrees(np.arctan(m1))


x_box = 8
y_box = 3
step = dist_pos /x_box
step_neg =dist_neg/y_box

for i in range(x_box):
    yg_1 = (lim_pos_up - (i)*step/np.cos(45*u.deg)) +  m*catal[:,7]
    # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
    yg_2 = (lim_pos_up - (i+1)*step/np.cos(45*u.deg)) +  m*catal[:,7]

    # ax.plot(catal[:,7],yg_1, color ='g')
    # ax.plot(catal[:,7],yg_2, color ='g')
    for j in range(y_box):
        yr_1 = (lim_neg_up - (j)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
        yr_2 = (lim_neg_up - (j+1)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
        good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
        
        np.savetxt(pruebas + 'subsec_%s/subsec_%s_%s_%s.txt'%(section, section, i, j)
                                    ,catal[good],fmt='%.7f %.7f %.4f %.4f %.4f %.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.0f %.0f %.0f %.0f %.5f %.5f %.5f %.5f %.5f %.3f',
                                    header ="'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
        
        
        
        ax.scatter(catal[:,7][good],catal[:,8][good],color =strin[np.random.choice(indices)])
        # ax.plot(catal[:,7],yr_1, color ='r')
        # ax.plot(catal[:,7],yr_2, color ='r')
# ax.plot(catal[:,7],yr_up, color ='r')
# ax.plot(catal[:,7],yr_down, color ='r')
# ax.plot(catal[:,7],yg_up, color ='g')
# ax.plot(catal[:,7],yg_down, color ='g')

ax.set_xlabel('x (130 mas/pix)')
ax.set_ylabel('y (130 mas/pix)')
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'section_B_Libr.png', dpi=300,bbox_inches='tight')

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
