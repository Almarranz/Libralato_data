#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:06:39 2022

@author: amartinez
"""

# =============================================================================
# Creates a .reg file in order to check cluster members in DS9
# =============================================================================




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable
from matplotlib import rcParams
import os
import glob
import sys
import math
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
# pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
pruebas = '/Users/amartinez/Desktop/PhD/HAWK/GNS_pm_scripts/clusters_regions/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
clusters = '/Users/amartinez/Desktop/PhD/Libralato_data/clusters_sections/'
name='WFC3IR'

# ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks_mean, dAks_mean, radio("),cluster_ID
# cluster ='/Users/amartinez/Desktop/morralla/GNS1_f7c4_dmu10_at_GNS2_f7c1_2022-12-16 18:46:37.762510/cluster_num2_2_knn10_area1.37/cluster3_0_0_knn_15_area_31094.89_pm_color.txt'
cluster = '/Users/amartinez/Desktop/morralla/Sec_B_dmu1.4_at_mean_kernnel_2023-01-24 11:30:54.148048/cluster_num0_0_knn30_area59.90/cluster0_0_0_knn_30_area_59.90_all_color.txt'
# cluster ='/Users/amartinez/Desktop/morralla/Sec_B_dmu1_at_minimun_2022-08-30/cluster_num32_2_knn10_area7.49/cluster32_0_0_knn_10_area_7.49_all_color.txt'
ID = np.loadtxt(cluster, unpack=True, usecols=(-1))
c_num = int(ID[0])

section = 'B'
# ra, dec, l, b, mul, mub, H, Ks,cluster = np.loadtxt(clusters + 'Sec_%s_WFC3IR_cluster%s_eps%s.txt'%(section, clus, eps),unpack=True)

# c_num = np.arange(0,1,1)
# eps_num = np.arange(0.290,0.500,0.01)

# for file_to_remove in glob.glob(pruebas + '*.reg'):
#     os.remove(file_to_remove)
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
ban_cluster = np.loadtxt(cata +'ban_cluster.txt')


ra, dec, l, b, mul, mub = np.loadtxt(cluster,unpack=True,usecols=(0,1,2,3,4,5))
# mul =(mul+5.72)
# mub = mub +0.2
with open(pruebas+ 'reg%s_clus%s_stars%s.reg'%(section, c_num,len(ra)), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'galactic'+'\n')
    f.close

# =============================================================================
# # with arrows
# for i in range(len(ra)):
#     with open(pruebas+ 'reg%s_clus%s.reg'%(section, c_num), 'a') as f:
#         print(mul[0],mub[0])
#         if mul[0]>0 and mub[0]>0:
#             f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180-math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
#             print('ssss')
#         elif mul[0]<0 and mub[0]>0:
#             f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,-1*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
#             print(mul[i],mub[i],-1*math.degrees(math.atan(mub[i]/mul[i])))
#         elif mul[0]>0 and mub[0]<0:
#             f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180 + (-1)*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
#             print('ssss')
#         elif mul[0]<0 and mub[0]<0:
#             print('ssss')
#             f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
# f.close
# =============================================================================
# without arrows
for i in range(len(ra)):
    with open(pruebas+ 'reg%s_clus%s_stars%s.reg'%(section, c_num,len(ra)), 'a') as f:
        print(mul[0],mub[0])
        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'\n')))   
        print('ssss')
f.close
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
clus_file ='/Users/amartinez/Desktop/PhD/regions/clusters/'
ra_ban, dec_ban = np.loadtxt(cata +'ban_cluster.txt', unpack = True)
with open(clus_file + 'ban_cluster.reg', 'w') as fb:
    fb.write('# Region file format: DS9 version 4.1'+"\n"+'global color = blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
    fb.close
for b in range(len(ra_ban)):
    with open(clus_file + 'ban_cluster.reg', 'a') as fb:
        fb.write('point(%s,%s) # point=x \n'%(float(ra_ban[b]),float(dec_ban[b])))
        
fb.close     
 


    
    
    
    