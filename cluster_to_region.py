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

# ra, dec, l, b, pml, pmb, H, Ks,cluster
c_num = [192]
eps_num= [0.122]
section = 'All'
# ra, dec, l, b, mul, mub, H, Ks,cluster = np.loadtxt(clusters + 'Sec_%s_WFC3IR_cluster%s_eps%s.txt'%(section, clus, eps),unpack=True)

# c_num = np.arange(0,1,1)
# eps_num = np.arange(0.290,0.500,0.01)

# for file_to_remove in glob.glob(pruebas + '*.reg'):
#     os.remove(file_to_remove)


for eps in eps_num:
    for clus in c_num:
        eps = round(eps,3)
        print(clus, eps)
        try:
            ra, dec, l, b, mul, mub, H, Ks,cluster = np.loadtxt(pruebas + 'Sec_%s_WFC3IR_cluster%s_eps%s.txt'%(section, clus, eps),unpack=True)
            with open(pruebas+ 'reg%s_clus%s_eps%s.reg'%(section, clus,eps), 'w') as f:
                f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'galactic'+'\n')
                f.close
                
            for i in range(len(ra)):
                with open(pruebas+ 'reg%s_clus%s_eps%s.reg'%(section, clus,eps), 'a') as f:

                    if mul[0]>0 and mub[0]>0:
                        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180-math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
                        print('ssss')
                    elif mul[0]<0 and mub[0]>0:
                        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,-1*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
                        print(mul[i],mub[i],-1*math.degrees(math.atan(mub[i]/mul[i])))
                    elif mul[0]>0 and mub[0]>0:
                        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,180 + (-1)*math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   
                        print('ssss')
                    elif mul[0]>0 and mub[0]>0:
                        print('ssss')
                        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),np.sqrt(mul[i]**2+mub[i]**2)*10,math.degrees(math.atan(mub[i]/mul[i]))),'\n')))   

        except:
            pass
# %%
print(mul[0],mub[0],-1*math.degrees(math.atan(mub[i]/mul[i])))
 


    
    
    
    