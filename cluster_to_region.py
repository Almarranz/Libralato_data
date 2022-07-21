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
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
clusters = '/Users/amartinez/Desktop/PhD/Libralato_data/clusters_sections/'
name='WFC3IR'

# ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks_mean, dAks_mean, radio("),cluster_ID
cluster ='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/Sec_A_clus/cluster_num1_0_knn9_area2.12/cluster1_1_4_knn_9_area_2.12.txt'
ID = np.loadtxt(cluster, unpack=True, usecols=(-1))
c_num = int(ID[0])

section = 'A'
# ra, dec, l, b, mul, mub, H, Ks,cluster = np.loadtxt(clusters + 'Sec_%s_WFC3IR_cluster%s_eps%s.txt'%(section, clus, eps),unpack=True)

# c_num = np.arange(0,1,1)
# eps_num = np.arange(0.290,0.500,0.01)

# for file_to_remove in glob.glob(pruebas + '*.reg'):
#     os.remove(file_to_remove)



ra, dec, l, b, mul, mub = np.loadtxt(cluster,unpack=True,usecols=(0,1,2,3,4,5))
mul =(mul+5.72)
mub = mub +0.2
with open(pruebas+ 'reg%s_clus%s.reg'%(section, c_num), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'galactic'+'\n')
    f.close
    
for i in range(len(ra)):
    with open(pruebas+ 'reg%s_clus%s.reg'%(section, c_num), 'a') as f:
        print(mul[0],mub[0])
        if mul[0]>0 and mub[0]>0:
            f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180-math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
            print('ssss')
        elif mul[0]<0 and mub[0]>0:
            f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,-1*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
            print(mul[i],mub[i],-1*math.degrees(math.atan(mub[i]/mul[i])))
        elif mul[0]>0 and mub[0]<0:
            f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180 + (-1)*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
            print('ssss')
        elif mul[0]<0 and mub[0]<0:
            print('ssss')
            f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
f.close
        
       
 


    
    
    
    