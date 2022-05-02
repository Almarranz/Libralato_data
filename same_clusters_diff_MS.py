#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:57:51 2022

@author: amartinez
"""

# =============================================================================
# Look for common elements in different cluster found in groups aroun different MS
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

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
search = 'dbs_'

trimmed_data='yes'

if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

all_cluster = []
for cluster_file in glob.glob(pruebas+'dbs_%scluster*.txt'%(pre)):
    each = np.loadtxt(cluster_file)
    for ele in range(len(each)):
        all_cluster.append(each[ele])
# %%
all_np = np.array(all_cluster)        

# %%
indices =[]
for e in range(len(all_np)):
# for e in range(2):
    ra, dec = all_np[e,0], all_np[e,1]
    new_all_np = np.delete(all_np,e,0)
    repeat = np.where((ra == new_all_np[:,0])&(dec == new_all_np[:,1]))
    print(len(repeat),len(repeat[0]))
    if len(repeat[0]>0):
        print(e, repeat[0])    
        for c in range(len(repeat)):
            print('Group %0.f, cluster %0.f and Group %0.f, cluster %0.f '
                  %(all_np[e][-2],all_np[e][-1],new_all_np[repeat][0][-2],new_all_np[repeat][-1][-1]))
            indices.append((all_np[e][-2],all_np[e][-1],new_all_np[repeat][c][-2],new_all_np[repeat][c][-1]))
    # %

# %%
indices_set = set(indices)
indices_np = np.array(list(indices_set))
sorted_array = indices_np[np.argsort(indices_np[:, 0])]
np.savetxt(pruebas + '%scommon_clusters.txt'%(search),sorted_array,fmt='%.0f',header = 'Group, cluster ---> Group, cluster') 
# %%
print(repeat[0])
    
    