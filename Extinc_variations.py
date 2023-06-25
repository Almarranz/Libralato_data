#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:25:17 2023

@author: amartinez
"""


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
from scipy.spatial import distance
import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib import cm
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# MS stars with cluster so far
# 14996	154855	954199	9192 	18538
# st_list = [14996	,154855,	954199,	9192,18538]
st_list = [14996,	154855,	954199,	9192,	18538,	1616,	14221,	14733,	17766,	18575,	18979,	611113,	612448,	16791,	538808,	987487,	1187124]
# st_list = [14996]1
maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
layer = 2
AKs = fits.open(maps + 'K%sHK_C.fit'%(layer),memmap=True)        
Ks_map = WCS(maps + 'K%sHK_C.fit'%(layer))
# pix_sc = AKs[0].header['CD2_2']
# pix_sc = AKs[0].header['CD2_1']
pix_sc = 20#pix/arcsec
sep = 200*pix_sc# in arcsec. Search area for extinction variation to look around
around = sep*2
with open(pruebas + 'st_map.reg','w') as fil:
    fil.write('# Region file format: DS9 version 4.1\nglobal color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nphysical\n')
for st in st_list:
    mas_ind = np.where(yso[:,2] == st)
    
    mas = yso[mas_ind]
    
    # cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
    
    
    
    cor = [mas[0][0:2][0],mas[0][0:2][1]]
    pix_Ks =   Ks_map.wcs_world2pix([cor],1)
    
    # Pixel scale roughlt correspont to 3 arsec every 60 pixels
    
    x_ar = np.arange(pix_Ks[0][1].astype(int) -around, pix_Ks[0][1].astype(int) + around+1,55)
    y_ar = np.arange(pix_Ks[0][0].astype(int) -around, pix_Ks[0][0].astype(int) + around+1,55)
    xv, yv = np.meshgrid(x_ar,y_ar)
    # %
    

    pos = np.vstack([yv.ravel(), xv.ravel()])
    dist=distance.cdist(np.array(pix_Ks),pos.T, 'euclidean')
    radio = np.where(dist < sep)
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(pos[0],pos[1], color = 'k',s =1)
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]], color = 'r',s =5)
    # ax.set_aspect('equal', 'box')
    # %
    AKs_clus = [AKs[0].data[pos[1][radio[1]][i]][pos[0][radio[1]][i]]
                for i in range(len(radio[1]))]
   
    print('st = %s\nMax diff = %.2f, std = %.2f, radio = %s'%(st, np.nanmax(AKs_clus) - np.nanmin(AKs_clus),np.nanstd(AKs_clus),sep/pix_sc))
    np_plot = np.array([pos[0][radio[1]], pos[1][radio[1]],AKs_clus]).T
# =============================================================================
#     with open(pruebas + 'st_map.reg','a') as fil:
#         fil.write('point(%s,%s) # point = x\n# text(%s,%s) text={%.3f}\ncircle(%s,%s,%s)\n'%(pix_Ks[0][0],pix_Ks[0][1],
#                                                                          pix_Ks[0][0],pix_Ks[0][1]-10,
#                                                                 AKs[0].data[pix_Ks[0][1].astype(int)][pix_Ks[0][0].astype(int)],
#                                                                          pix_Ks[0][0],pix_Ks[0][1],sep))
# 
# =============================================================================
    
# %%   
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    # ax = fig.add_subplot(projection='3d')
    ax[0].set_title('%s'%(st))
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]],c = np.log(np.array(AKs_clus)))
    AKs_c = np.array(AKs_clus)
    ax[0].scatter(pos[0][radio[1]], pos[1][radio[1]],c=AKs_clus,norm=colors.LogNorm(vmin=np.nanmin(AKs_c), vmax=np.nanmax(AKs_c)))

    ax[0].set_aspect('equal', 'box')
    
    np_plot = np.array([pos[0][radio[1]], pos[1][radio[1]],AKs_clus]).T
    
    
    
    
    x = np_plot[:, 0]  # X coordinates (first column)
    y = np_plot[:, 1]  # Y coordinates (second column)
    z = np_plot[:, 2]  # Z values (third column)
    
    buenos = np.where(np.isnan(z) == False)
    x = x[buenos]
    y = y[buenos]
    z = z[buenos]
    
    
    
    # Create a grid for contour plotting
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    
    # Perform interpolation to obtain Z values on the grid
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='nearest')
    
    # Create a contour plot
    
    v_min, v_max = 2,4
    lev =np.arange(v_min,v_max+1,1)
    ax[1].tricontour(x, y, z, levels=lev, linewidths=2,colors = 'white',linestyles='dashed',vmin=v_min, vmax=v_max)  # Contour lines
    axcb = ax[1].tricontourf(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), levels=lev,vmin=v_min, vmax=v_max, cmap='viridis')  # Filled contours
    ax[1].set_aspect('equal', 'box')
    fig.colorbar(axcb,ax =ax[1],shrink=0.7)
    
# %%

    x_ = np_plot[:,0]
    y_ = np_plot[:,1]
    z_ = np_plot[:,2]
    
    buenos = np.where(np.isnan(z_) == False)
    x_ = x_[buenos]
    y_ = y_[buenos]
    z_ = z_[buenos]
    
    center_x = pix_Ks[0][0]
    center_y = pix_Ks[0][1]
    distances = np.sqrt((x_ - center_x) ** 2 + (y_ - center_y) ** 2)
    
    perc = np.arange(0.1,1.2,0.2)
    dist_lim = np.nanmax(distances)*perc
    std_devs = [np.nanstd(z_[np.where(distances < dl)]) for dl in dist_lim]
    
    std_levels =np.array([])
    marked = np.zeros(len(distances))
    
    std_levels = np.zeros(len(distances))
    for i, dl in enumerate(dist_lim):
        print(len(distances))
        under = np.where((distances <=dl) & (std_levels ==0))
        std_levels[under] = std_devs[i]
       
    
    fig, ax = plt.subplots(1,2,figsize = (14,7))
    ra_dec =   Ks_map.wcs_pix2world(np.array([x_,y_]).T,1)
    # im = ax[1].scatter(x_,y_, c=std_levels,s=100, cmap = 'inferno', vmin = 0, vmax = 0.3)
    im = ax[1].scatter(ra_dec[:,0],ra_dec[:,1], c=std_levels,s=100, cmap = 'viridis', vmin = 0, vmax = 0.3)
    fig.colorbar(im)
   
    std_devs = []
    
    unique_distances, ind_t = np.unique(np.round(distances,0),return_index=True)
    # unique_distances = distances
    for dist in unique_distances:
        indices = np.where(distances < dist)
        std_dev = np.nanstd(z_[indices])
        std_devs.append(std_dev)
    std_devs =np.array(std_devs)
    # good_xy = np.where(np.round()) 
    
    d_line  = np.where(std_devs>np.nanmean(std_devs))[0][0]
    unique_distances, ind_t = np.unique(np.round(distances,0),return_index=True)
    
    ax[0].set_ylim(0,0.35)
    ax[0].set_title(st)
    ax[0].axhline(np.nanmean(std_devs),color = 'red')
    ax[0].axvline(unique_distances[d_line],linestyle='dashed',color ='red')
    ax[0].text(unique_distances[d_line], 0.10,'%s'%(unique_distances[d_line]) )
    ax[0].scatter(unique_distances,std_devs)
    
   
    # circle1 = plt.Circle((center_x, center_y),unique_distances[d_line] , facecolor ='none', edgecolor = 'k')
    # ax[1].add_patch(circle1)
    
    rad_cir = (unique_distances[d_line]/20)*u.arcsec.to(u.degree)
    ax[1].set_title('$\\sigma AKs < \\overline{\\sigma AKs}$ --> %.1f arcsec'%(rad_cir*3600))
    ax[0].set_title('%.1f arcsec'%(sep/20))
    circle1 = plt.Circle((cor[0], cor[1]),rad_cir , facecolor ='none', edgecolor = 'red')
    
    ax[1].add_patch(circle1)
# %%
    
    # ra_dec =   Ks_map.wcs_pix2world(np.array([x_,y_]).T,1)
    
   





