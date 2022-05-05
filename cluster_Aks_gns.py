#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:19:41 2022

@author: amartinez
"""

# =============================================================================
# Here we are going to extract the value of AKs for each member of the selected 
# cluster obtained with dbscan_coparation.py or hdbscan_comparation.py, that we 
# will use to construct a cluster with SPISEA
# =============================================================================%%
# %%
import numpy as np
from astropy.coordinates import match_coordinates_sky
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
# %%
name='WFC3IR'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
# %%

# RAdeg 0, e_RAdeg 1, DEdeg 2, e_DEdeg 3, RAJdeg 4,e_RAJdeg 5, DEJdeg6, e_DEJdeg 7, 
# RAHdeg 8, e_RAHdeg 9, DEHdeg 10, e_DEHdeg 11,  RAKsdeg 12, e_RAKsdeg 13,
# DEKsdeg 14, e_DEKsdeg 15, J 16, eJ 17, H 18, eH 19, Ks 20, eKs 21,
# iJ 22, iH 23, iKs 24
# =============================================================================
# Aks_gns = np.loadtxt(gns_ext + 'cen-nd.txt')
# =============================================================================

# RAdeg 0, e_RAdeg 1, DEdeg 2, e_DEdeg 3, J 4, eJ 5, H 6, eH 7, Ks 8, eKs 9
# FJH 10, FHK 11, AJ1JH 12, e_AJ1JH 13,  AH1JH 14, e_AH1JH 15, 
# AH1HK 16, e_AH1HK 17, AK1HK 18, e_AK1HK 19, AH2HK 20, e_AH2HK 21, 
# AK2HK 22, e_AK2HK 23  

Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %%
AKs_np = Aks_gns.to_numpy()
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)
AKs_center =AKs_np[center]

# %%
clu_gr = [4,1]
search = 'dbs'
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree)
with open(pruebas +'AKs_%s_clusters.txt'%(search),'w') as file:
    file.write('# mean AKs, group, cluster')
for g in range(95):
    for c in range(6):
        
        try:
            clus_L = np.loadtxt(pruebas + '%s_cluster%s_of_group%s.txt'%(search,c,g))
            print('%s_cluster%s_of_group%s.txt'%(search,g,c))
            
            clus_coord =  SkyCoord(ra=clus_L[:,0]*u.degree, dec=clus_L[:,1]*u.degree)
            
            idx = clus_coord.match_to_catalog_sky(gns_coord)
            gns_match = AKs_center[idx[0]]
            good = np.where(gns_match[:,11] == -1)
            if len(good[0]) != len(gns_match[:,11]):
                print('%s foreground stars in this cluster'%(len(gns_match[:,11]) - len(good)))
            gns_match_good = gns_match[good]
            AKs_clus = [float(gns_match_good[i,18]) for i in range(len(gns_match_good[:,18]))]
            with open(pruebas +'AKs_%s_clusters.txt'%(search),'a') as file:
                file. write('\n'+'%.2f %.2f %s %s'%(np.mean(AKs_clus), np.std(AKs_clus), g, c))
                #file.close
        except:
            pass
            










