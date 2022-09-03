#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:00:25 2022

@author: alvaromartinez
"""
# =============================================================================
# Here we are going to plot the some data for the simulated clusters
# =============================================================================
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from matplotlib.ticker import FormatStrFormatter

import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from sklearn.preprocessing import StandardScaler

from scipy.stats import gaussian_kde

import astropy.coordinates as ap_coor
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
pruebas='/Users/amartinez/Desktop/Phd/Libralato_data/pruebas/'
sim_dir ='/Users/amartinez/Desktop/PhD/Libralato_data/simulated_stat/'
# Load the data from the desired subregion
dmu_lim = 1
area = 7.5
section ='B'
sub_sec = '0_0' 
bucles = 10000
simulated_by = 'kern'

mean_values = np.loadtxt(sim_dir + 'sec%s_%s_std_area%s_dmul%s_%sims_%s.txt'%(section,sub_sec,
                                                             area,dmu_lim,bucles,simulated_by))
sigm_values = np.loadtxt(sim_dir + 'sig_sec%s_%s_std_area%s_dmul%s_%sims_%s.txt'%(section,sub_sec,
                                                             area,dmu_lim,bucles,simulated_by))

# %%
# 1
fig, ax = plt.subplots(1,2, figsize=(20,10))
gra = 0
eje = 0
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\mu}_{l}$',fontsize = 30)

gra = 1
eje = 1
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\mu}_{b}$',fontsize = 30)
# %
# 2
fig, ax = plt.subplots(1,2, figsize=(20,10))
gra = 0
eje = 2
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Ra',fontsize = 30)

gra = 1
eje = 3
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Dec',fontsize = 30)
# %
# 3
fig, ax = plt.subplots(1,2, figsize=(20,10))

gra = 0
eje = 4
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Color (H - Ks)',fontsize = 30)

# %
# 4
fig, ax = plt.subplots(1,2, figsize=(20,10))


gra = 0
eje = 0
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(sigm_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(sigm_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]-np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]+np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\sigma}_{l}$',fontsize = 30)

gra = 1
eje = 1
ax[gra].hist(sigm_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(sigm_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]-np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]+np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\sigma}_{b}$',fontsize = 30)
      


# %%
print



