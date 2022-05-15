#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:25:06 2022

@author: amartinez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:43:50 2022

@author: amartinez
"""
# =============================================================================
# This script selects the data within a certain uncertainty and at the galactic
# center and dived it into 4 sections.
# =============================================================================

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
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import astropy.coordinates as ap_coor

# %%

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
    
# %%
# We upload galactic center stars, that we will use in the CMD
# catal=np.loadtxt(results+'refined_%s_PM.txt'%(name))
# catal_df=pd.read_csv(pruebas+'%s_refined_with_GNS_partner_mag_K_H.txt'%(name),sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'])

# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
# center_definition='G_G'#this variable can be L_G or G_G
# if center_definition =='L_G':
#     valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
#     catal=catal[valid]
#     center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
# elif center_definition =='G_G':
#     valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
#     catal=catal[valid]
#     center=np.where(catal[:,3]-catal[:,4]>1.3)
# catal=catal[center]
# dmu_lim = 0.5
# vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
# catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))
# %%
# This are the slopes of the lines used for dividing the data
m = -0.8
m1 = 1

# %
# =============================================================================
# HEre we are divied the data in sections
# =============================================================================
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)
# =============================================================================
# Section A
# =============================================================================

yr_A = 32700 + m*catal[:,7]
yb_Adown = 18500 + m1*catal[:,7]

sec_A = np.where((catal[:,8]<yr_A) & (catal[:,8]>yb_Adown))
ax.scatter(catal[:,7][sec_A],catal[:,8][sec_A],color = 'b', alpha = 0.3)


# =============================================================================
# Section B
# =============================================================================
# fig, ax = plt.subplots(1,1,figsize=(10,10))
yr_B = 32650 + m*catal[:,7]
yb_Bup = 24000 + m1*catal[:,7]
yb_Bdown = -1000 + m1*catal[:,7]

sec_B = np.where((catal[:,8]>yr_B) & (catal[:,8]>yb_Bdown) &(catal[:,8]<yb_Bup))
ax.scatter(catal[:,7][sec_B],catal[:,8][sec_B],color = 'red', alpha = 0.3)

# ax.set_xlim(6000,25000)
# ax.set_ylim(17000,40000)

# %
# =============================================================================
# Section C
# =============================================================================
# fig, ax = plt.subplots(1,1,figsize=(10,10))
# ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)

yr_Cup = 32600 + m*catal[:,7]
yr_Cdown = 23500 + m*catal[:,7]
yb_Cup = 1000 + m1*catal[:,7]
sec_C = np.where((catal[:,8]<yr_Cup) & (catal[:,8]>yr_Cdown) & (catal[:,8]<yb_Cup) )
ax.scatter(catal[:,7][sec_C],catal[:,8][sec_C],color = 'green', alpha = 0.3)




# %
# =============================================================================
# Section_D
# =============================================================================

# fig, ax = plt.subplots(1,1,figsize=(10,10))
# ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)

yr_Dup = 23500 + m*catal[:,7]
yb_Dup = 1000 + m1*catal[:,7]
sec_D = np.where((catal[:,8]<yr_Dup) & (catal[:,8]<yb_Dup) )
ax.scatter(catal[:,7][sec_D],catal[:,8][sec_D],color = 'orange', alpha = 0.3)

# ax.set_xlim(2000,8000)
# ax.set_ylim(25000,30000)
# %%

sections =[sec_A,sec_B,sec_C,sec_D]
sec_names = ['sec_A','sec_B','sec_C','sec_D']
data_len = 0
for s in range(len(sections)):
    np.savetxt(results + '%s_%smatch_GNS_and_%s_refined_galactic.txt'%(sec_names[s],pre,name),catal[sections[s]],
               fmt='%.7f %.7f %.4f %.4f %.4f %.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.0f %.0f %.0f %.0f %.5f %.5f %.5f %.5f %.5f %.3f',
               header ="'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
    data_len += len(catal[sections[s]])
print(data_len,len(catal))
if len(catal) != data_len:
    frase = 'YOU ARE MISSING %s POINTS'%(data_len - len(catal))
    print('\n'.join((len(frase)*'*',frase,len(frase)*'*')))
# %








