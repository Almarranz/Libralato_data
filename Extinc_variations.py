#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:25:17 2023

@author: amartinez
"""

# The idea is to check the variations around the massive stars where we have
# find a cluster to justify the search areas around the massive stars.
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
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from extinction_simulation import extinction_now
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'

yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# MS stars with cluster so far
# 14996	154855	954199	9192 	18538

mas_ind = np.where(yso[:,2] == 14996)

mas = yso[mas_ind]

maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
# cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'

AKs = fits.open(maps + 'K2HK_C.fit',memmap=True)
AH = fits.open(maps + 'H2HK_C.fit',memmap=True)

H_map = WCS(maps +  'H2HK_C.fit')
Ks_map = WCS(maps + 'K2HK_C.fit')

cor = [266.3864974,-28.9379862]
pix_Ks =   Ks_map.wcs_world2pix([cor],1)

x_ar = np.arange(pix_Ks[0][1].astype(int) -10,pix_Ks[0][1].astype(int) +11,1)
y_ar = np.arange(pix_Ks[0][0].astype(int) -10,pix_Ks[0][0].astype(int) +11,1)
xv, yv = np.meshgrid(x_ar,y_ar)
# %%
fig, ax = plt.subplots(1,1)
ax.scatter(xv, yv)
# %%
around = np.where()
AKs_clus =AKs[0].data[pix_Ks[0][1].astype(int)][pix_Ks[0][0].astype(int)]







