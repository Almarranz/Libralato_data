#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:37:18 2023

@author: amartinez
"""
# MS selection and clustering in the similar way it was previouly done by
# co_mouving_groups.py and dbscan_compartion.py but in a single file

# Firts we select the MS from the lists of Dong et al. that have a counterpary
# in Libralato catalog. Then we will run dbscan around it.

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
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# 
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
# name='ACSWFC'
trimmed_data='no'
only_match = 'yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
#ra, dec, ID(in ACSWFC_PM or WFC3IR_PM),Original list, Altervative Id
yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# "'RA_gns' 0	DE_gns' 1	Jmag' 2	Hmag' 3	Ksmag' 4	ra' 5	dec' 6	x_c' 7	y_c' 8	
# mua' 9	dmua' 10	mud' 11	dmud' 12	time' 13	n1' 14	n2' 15	ID' 16	mul' 17	mub' 18	
# dmul' 19	dmub' 20	m139' 21	Separation'" 22			
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))

# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>3) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 2
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]
# mag_cut = np.where((catal[:,-2] > 13) & ((catal[:,-2] < 160)))# this magnitude cut is maden by Libralato et al. 
# catal = catal[mag_cut]
# %%
catal[:,17] =catal[:,17]*-1 - 5.74
catal[:,18] = catal[:,18]-0.20
# %%

ms_coord = SkyCoord(ra = yso[:,0], dec = yso[:,1], unit = 'degree')
lib_coord = SkyCoord(ra = catal[:,5], dec = catal[:,6], unit = 'degree')

max_sep = 0.05*u.arcsec # the radio of 
idx,d2d,d3d = ms_coord.match_to_catalog_sky(lib_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
ms_match = yso[sep_constraint]
tipo_match = tipo[sep_constraint]
lib_match = catal[idx[sep_constraint]]


# %%
# REGIONS ZONE
# =============================================================================
# # generates a region file with arrows for the massive stars from wich we have pms.
# with open(pruebas+ 'pm_of_Ms_in_%s.reg'%(name), 'w') as f:
#          f.write('# Region file format: DS9 version 4.1\nglobal color=cyan dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\ngalactic')
# 
# 
#          f.close
#          
# # "'RA_gns' 0	DE_gns' 1	Jmag' 2	Hmag' 3	Ksmag' 4	 ra' 5	dec' 6	x_c' 7	y_c' 8	
# # mua' 9	 dmua' 10	mud' 11	dmud' 12	time' 13	n1' 14	n2' 15	ID' 16	mul' 17	mub' 18	
# # dmul' 19	dmub' 20	m139' 21	Separation'" 22		
# 
# # a, b = 9,11
# a, b = 17,18
# 
# for s in range(len(lib_match)):
#     with open(pruebas+ 'pm_of_Ms_in_%s.reg'%(name), 'a') as f:   
#         lib_gal = SkyCoord(ra = lib_match[s][0],dec = lib_match[s][1], unit ='degree').galactic
#         f.write('\n'.join(('\npoint(%s,%s) # point=x'%(lib_gal.l.value,lib_gal.b.value),'# vector(%s,%s,%s",%s)'%(lib_gal.l.value,lib_gal.b.value,
#                                                                                                                 np.sqrt(lib_match[s][a]**2+lib_match[s][b]**2)*20,
#                                                                                                                 np.degrees(np.arctan2(lib_match[s][b],lib_match[s][a])))
#                            ,'# text(%s,%s) text={%.2f}'%(lib_gal.l.value,lib_gal.b.value,lib_match[s][16]))))  
# 
# 
# clark = ascii.read(cata + 'clark_2018_.dat')
# 
# 
# 
# ra_clark = []
# dec_clark = []
# 
# for i in range(len(clark)):
#     ra_clark.append('%sh%sm%s'%(clark['RAh'][i],clark['RAm'][i],clark['RAs'][i]))
#     dec_clark.append('%sh%sm%s'%(clark['DEd'][i],clark['DEm'][i],clark['DEs'][i]))
# 
# 
# with open(pruebas+ 'MS_Clark.reg', 'w') as f:
#          f.write('# Region file format: DS9 version 4.1\nglobal color=pink dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5')
# 
# 
#          f.close
#          
# with open(pruebas+ 'MS_Clark.reg', 'a') as f:
#     for i in range(len(clark)):
#         f.write('\npoint(%s:%s:%s,-%s:%s:%s) # point=x'%(clark['RAh'][i],clark['RAm'][i],clark['RAs'][i],clark['DEd'][i],clark['DEm'][i],clark['DEs'][i]))
# =============================================================================
# %%

# =============================================================================
# # Analisys for the Ms stars.
# ms_sl, ms_sb = np.std(lib_match[:,17]),  np.std(lib_match[:,18])
# ms_l, ms_b = np.mean(lib_match[:,17]),  np.mean(lib_match[:,18])
# #
# ms_E = np.where(lib_match[:,17]<0)
# all_E = np.where(catal[:,17]<0)
# ms_E_sig =  np.std(lib_match[:,17][ms_E])
# catal_E_sig = np.std(catal[:,17][all_E])
# =============================================================================

# %%
# clus_sizes = ascii.read(cata + 'Arches_Quintuplet_sizes.dat')
# %%
r_u = 50*u.arcsec
from dbscan_GC import dbscan_GC as cluster_search
import time


# list_clus= ['all_color','all','vel_col']
list_clus= ['vel']

for clus_by in list_clus:
    star_n=0
    tic = np.datetime64('now')
    # for j in range(len(ms_match)):
    for j in range(1,6):
        star_n +=1
        ms_match_c =  SkyCoord(ra = [ms_match[j][0]], dec = [ms_match[j][1]], unit = 'degree')
        idxc, group, d2d,d3d = lib_coord.search_around_sky(ms_match_c,r_u)
       
        
    
        # "'RA_gns' 0	DE_gns' 1	Jmag' 2	Hmag' 3	Ksmag' 4	ra' 5	dec' 6	x_c' 7	y_c' 8	
        # mua' 9	dmua' 10	mud' 11	dmud' 12	time' 13	n1' 14	n2' 15	ID' 16	mul' 17	mub' 18	
        # dmul' 19	dmub' 20	m139' 21	Separation'" 22		
        
        # dbscan_GC(pmra, pmdec, x, y,Ra,Dec, color_A, color_B, clustered_by, samples_dist, Ms_ra, Ms_dec)
        something = cluster_search(catal[:,9][group], catal[:,11][group], catal[:,7][group], catal[:,8][group], 
                       catal[:,0][group], catal[:,1][group],catal[:,3][group], catal[:,4][group], clus_by, 20,
                       lib_match[j])
        if type(something) is int:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.scatter(catal[:,0], catal[:,1], color = 'k', alpha = 0.01)
            ax.scatter(catal[:,0][group], catal[:,1][group])
            ax.scatter(ms_match_c.ra, ms_match_c.dec, s =10, label = '%s, %s'%(star_n,tipo_match[j]))
            ax.invert_xaxis()
            ax.legend()
        
    
        print(star_n, type(something))
    toc =  np.datetime64('now')
    
    tic_toc = toc - tic
    print('%s took %s'%(clus_by, (tic_toc)))

