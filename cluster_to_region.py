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
clus = 0
eps= 0.199
section = 'A'
ra, dec, l, b, mul, mub, H, Ks,cluster = np.loadtxt(clusters + 'Sec_%s_WFC3IR_cluster%s_eps%s.txt'%(section, clus, eps),unpack=True)


with open(pruebas+ 'reg%s_clus%s_eps%s.reg'%(section, clus,eps), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'galactic'+'\n')
    f.close
    
for i in range(len(ra)):
    with open(pruebas+ 'reg%s_clus%s_eps%s.reg'%(section, clus,eps), 'a') as f:
        
    
        f.write('\n'.join(('point(%s,%s) # point=x'%(float(l[i]),float(b[i])),'# vector(%s,%s,%s",%s)'%(float(l[i]),float(b[i]),mul[i]*10,math.degrees(math.atan(mub[i]/mul[i]))),'\n')))       
   


 


    
    
    
    