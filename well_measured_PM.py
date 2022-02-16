#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:13:44 2022

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
import scipy.integrate as integrate
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter

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


# %%

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'

# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
# mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms
# mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms = np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True)
ep1_fix=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,1))
ep2_fix=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,2))
mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )
all_ep1= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=False)


# %%
ep12,ep12_ind,ep21_ind=np.intersect1d(ep1_fix[:,9],ep2_fix[:,9], return_indices=True,assume_unique=True)
# You have to use these index with the whole catalog of 812377 stars, not witg tge well mesuared ones
# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
pm_wmp=catal[ep12.astype(int)]#pm_wmp stands for proper motion well mesaured photometry
all_eps=all_ep1[ep12.astype(int)]
# Discarted stars with errors equals 99.999mas/yr
# Discarted stars with velocities >70

velocity=np.sqrt(pm_wmp[:,4]**2+pm_wmp[:,6]**2)
v_valid=np.where((pm_wmp[:,5]<90) & (velocity<70) )
pm_wmp=pm_wmp[v_valid]
all_eps=all_eps[v_valid]
# %%
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(mag,catal[:,5],s=0.1,color='k',alpha=1)
ax.scatter(all_eps[:,0],pm_wmp[:,5],s=0.1,color='red',alpha=1)
ax.set_ylim(0,10)
ax.set_xlim(12,24)
# %%
# Condiytions for the proper motions:
#     (a) Pm uncertainty better that 85th percentile for any given magnitude
#     (b) PM uncertainty smaller than 0.7 also good, regardless of the percentile
paso=0.1
a=np.round(min(all_eps[:,0]))
b=np.round(max(all_eps[:,0]))
mag_b=np.digitize(all_eps[:,0], np.arange(a,b,paso), right=True)
xy_error=np.sqrt(pm_wmp[:,5]**2 + pm_wmp[:,7]**2)# I have combinned both axis error, I have the impresion thar libralato applies conditions for each axis separetly
error_valid=[]
all_sum=[]
# for i in range(len(np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),1))+1):#that last +1 if for picking up the last bin
for i in range(min(mag_b),(max(mag_b)+1)):
    try:
        mag_binned=np.where(mag_b==i)
        error_i=xy_error[mag_binned]
        print('%.5f'%(np.percentile(error_i,85)),i,len(error_i),len(mag_binned[0]))
        perc = np.percentile(error_i,85)
        all_sum.append(len(error_i))
        for j in range(len(error_i)):
            if error_i[j] < 10:
                if (error_i[j] <= 0.1) or (error_i[j] <= perc):
                    error_valid.append(mag_binned[0][j]) 
    except :
        print('Fallo:',len(error_i))
    
print(sum(all_sum),'All after a')
print(len(error_valid))
pm_wmp=pm_wmp[error_valid]
all_eps=all_eps[error_valid]
print(len(pm_wmp),'Condition b')

# %%
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(mag,catal[:,5],s=0.1,color='k',alpha=1)
ax.scatter(all_eps[:,0],pm_wmp[:,5],s=0.1,color='red',alpha=1)
ax.set_ylim(0,10)
ax.set_xlim(12,24)

np.savetxt(pruebas+'refined_%s_PM.txt'%(name),pm_wmp)
np.savetxt(pruebas+'refined_%s_phot.txt'%(name),all_eps)









