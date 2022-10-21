#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:02:55 2022

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
# %%plotiin parametres
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
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})

lim_dmul = 2
article = '/Users/amartinez/Desktop/PhD/My_papers/Libralato/'
im1 = plt.imread(article+'nsd_limdmu_b%s.png'%(lim_dmul))
im2 = plt.imread(article+'nsd_limdmu_l%s.png'%(lim_dmul))

ims=[im1,im2]
fig, ax = plt.subplots(1,2,figsize=(20,10))
for i in range(2):
        ax[i].imshow(ims[i])
        ax[i].axis('off')
plt.subplots_adjust(wspace=-0.005, hspace=0)
# plt.savefig(article + 'nsp_mub_mul.png', bbox_inches='tight',dpi=300)    

# %%
# Making a .reg file for the young massive stars in Libralato.
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

with open(article + 'young_stars.reg', 'w') as file:
    file.write('# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n')

#mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
ms_ra, ms_dec =np.loadtxt(pruebas +'pm_of_Ms_in_WFC3IR.txt', unpack = True, usecols = (4,5))# this are the information (pm, coordinates and ID) for the Ms that remain in the data after triming it 

for i in range(len(ms_ra)):
    with open(article + 'young_stars.reg', 'a') as file:
        file.write('circle(%s,%s,12.000")#fill=1 color=blue\n'%(ms_ra[i],ms_dec[i]))

# %%
if True:
    print('youmama')







