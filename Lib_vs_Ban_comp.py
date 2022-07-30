#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 19:22:28 2022

@author: amartinez
"""
# =============================================================================
# Here we are going to compare a bunch of cluster files with each other and see
# if any of then are closer than a certain distance. The idea is to compare
# cluster from Libralato and Bans catalogs
# 
# ============================================================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.coordinates as ap_coor
import pandas as pd
# import spisea
# from spisea import synthetic, evolution, atmospheres, reddening, ifmr
# from spisea.imf import imf, multiplicity
# from astropy.stats import sigma_clip
from astropy.coordinates import FK5
import time
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

section = 'A'
catal=np.loadtxt(results + 'sec_%s_match_GNS_and_WFC3IR_refined_galactic.txt'%(section))
# Path to the folder with the folders cointaining the clusters 
lib_files = '/Users/amartinez/Desktop/morralla/combined_Sec_A_clus/'
ban_files = '/Users/amartinez/Desktop/morralla/BAN_combined_Sec_A_clus/'
tic = time.perf_counter()
color_lib ='lime'
colo_ban = 'fuchsia'
for lib_folder in sorted(glob.glob(lib_files + 'cluster_num*'),key = os.path.getmtime):
    # print(30*'-')
    # print(os.path.basename(lib_folder))
    for file in glob.glob(lib_folder + '/cluster*'):
        lib_ra, lib_dec = np.loadtxt(file,unpack=True, usecols = (0,1))
        lib_coord = SkyCoord(ra = lib_ra, dec = lib_dec,unit = 'deg',frame = 'icrs',equinox ='J2000',obstime='J2014.2')
        for ban_folder in sorted(glob.glob(ban_files + 'ban_*'),key = os.path.getmtime):
            # print(os.path.basename(ban_folder))
            for file1 in glob.glob(ban_folder + '/ban_*'):
                ban_ra, ban_dec = np.loadtxt(file1, unpack = True, usecols = (0,1))
                ban_coord = SkyCoord(ra = ban_ra*u.deg, dec = ban_dec*u.deg,frame = 'icrs',equinox ='J2000')
                
                idx, d2d, dd3 = ban_coord.match_to_catalog_sky(lib_coord)
                min_dist = min(d2d.value)
                if min_dist < 1/3600:
                    fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # ra, dec, l, b, pml, pmb,J, H, Ks,x, y, AKs_mean, dAks_mean, radio("),cluster_ID
                    lib_clus = np.loadtxt(file)
                    # ra, dec, l, b, pml, pmb,J, H, Ks, AKs_mean, dAks_mean, radio("),cluster_ID
                    ban_clus = np.loadtxt(file1)
                    ax[0].scatter(catal[:,17],catal[:,18],color = 'k',alpha = 0.01)
                    ax[0].scatter(lib_clus[:,4]+5.78,lib_clus[:,5],color = color_lib)
                    ax[0].scatter(ban_clus[:,4],ban_clus[:,5],color = colo_ban)
                    ax[0].set_xlim(-10,10)
                    ax[0].set_ylim(-10,10)
                    ax[0].invert_xaxis()
                    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
                    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
                    
                    ax[1].scatter(catal[:,0],catal[:,1],color = 'k',alpha = 0.01)
                    ax[1].scatter(lib_clus[:,0],lib_clus[:,1],color = color_lib)
                    ax[1].scatter(ban_clus[:,0],ban_clus[:,1],color = colo_ban)
                    ax[1].set_xlabel('Ra(deg)',fontsize =30) 
                    ax[1].set_ylabel('Dec(deg)',fontsize =30) 
                    
                    ax[2].scatter(catal[:,3]-catal[:,4],catal[:,4],color = 'k',alpha = 0.01)
                    ax[2].scatter(lib_clus[:,7]-lib_clus[:,8],lib_clus[:,8],color = color_lib)
                    ax[2].scatter(ban_clus[:,7]-ban_clus[:,8],ban_clus[:,8],color = colo_ban)
                    ax[2].set_xlabel('H - Ks',fontsize =30) 
                    ax[2].set_ylabel('Ks',fontsize =30) 
                    ax[2].invert_yaxis()
                    ax[2].set_xlim(1.3,3)
                    
                    fig.suptitle('Lib: %s <------> Ban:%s'%(os.path.basename(file),os.path.basename(file1)))
                    print(50*'+')
                    print(os.path.basename(file),os.path.basename(file1))
                    print(50*'+')
                    plt.show()
                    # sys.exit('97')
                # sys.exit()
    print('Donde with folder: %s'%(os.path.basename(lib_folder)))            
toc = time.perf_counter()
print('Comparison took: %.1f sec'%(toc-tic))

                    