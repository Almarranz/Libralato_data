#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:45:30 2023

@author: amartinez
"""

# We are goin to desextinguir the stars in the Arches (or Quintuplet) using 
# Paco´s extinction maps and then simulate the extinction magnitudes of those
# stars in other places of the galactic center (using the same extinction maps)
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
from astropy import wcs
from astropy.wcs import WCS
import pandas as pd
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
# clus_now: the Arches(Quintuplet) cluster as recovered by dbscan 
# clus_future: the Arches after evolution.
def extinction_sim(clus_now, clus_future):
    maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
    # cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
    
    AKs = fits.open(maps + 'K2HK_C.fit',memmap=True)
    AH = fits.open(maps + 'H2HK_C.fit',memmap=True)

    H_map = WCS(maps +  'H2HK_C.fit')
    Ks_map = WCS(maps + 'K2HK_C.fit')
    ra_dec =[[clus_now[i][0],clus_now[i][1]] for i in range(len(clus_now))]
    
    pix_H =   H_map.wcs_world2pix(ra_dec,1)
    pix_Ks =  Ks_map.wcs_world2pix(ra_dec,1)
    
    AH_clus =[AH[0].data[pix_H[i][1].astype(int)][pix_H[i][0].astype(int)] for i in range(len(pix_H))]
    AKs_clus =[AKs[0].data[pix_Ks[i][1].astype(int)][pix_Ks[i][0].astype(int)] for i in range(len(pix_Ks))]
    
    # %%
    
    clus_future[:,-2] -= AH_clus
    clus_future[:,-1] -= AKs_clus
    
    ra_dec_f =[[clus_future[i][0],clus_future[i][1]] for i in range(len(clus_future))]
    
    pix_H_f =   H_map.wcs_world2pix(ra_dec_f,1)
    pix_Ks_f =  Ks_map.wcs_world2pix(ra_dec_f,1)
    
    AH_clus_f =[]
    AKs_clus_f =[]
    for i in range(len(pix_H_f)):
        try:
            AH_clus_f.append(AH[0].data[pix_H_f[i][1].astype(int)][pix_H_f[i][0].astype(int)])
            AKs_clus_f.append(AKs[0].data[pix_Ks_f[i][1].astype(int)][pix_Ks_f[i][0].astype(int)])
        except:
            AH_clus_f.append(99)
            AKs_clus_f.append(99)
            print('This is out of the extinction map bouns:%.3f,%.3f'%(clus_future[i][0],clus_future[i][1]))
            continue
    # AH_clus_f = [AH[0].data[pix_H_f[i][1].astype(int)][pix_H_f[i][0].astype(int)] for i in range(len(pix_H_f))]
    # AKs_clus_f =[AKs[0].data[pix_Ks_f[i][1].astype(int)][pix_Ks_f[i][0].astype(int)] for i in range(len(pix_Ks_f))]
    
    clus_future[:,-2] += AH_clus_f
    clus_future[:,-1] += AKs_clus_f
    
    return clus_future










