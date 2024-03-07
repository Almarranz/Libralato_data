#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:19:14 2024

@author: amartinez
"""

# =============================================================================
# When calcualting the absolute pm for the Quintuple cluster I obtained very 
# similar values for mu_ra and mu_dec of those of Hosek et al 2022. But then,
# whe comparing all the stars in the field I got shit. This to compare just the
# value of the member of the cluster and figure out what is going on
# =============================================================================
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
import species
import random
from astropy.stats import sigma_clip
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from texttable import Texttable
import latextable
from scipy.optimize import curve_fit
from scipy.special import erf 
import IPython
import astroalign as aa
from astropy.time import Time
import copy
from compar_lists import compare_lists
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
# Enable automatic plotting mode
IPython.get_ipython().run_line_magic('matplotlib', 'auto')
# IPython.get_ipython().run_line_magic('matplotlib', 'inline')

# Here we are gooing to look for the Arches cluster using dbscan
# %% Here we are going to trimm the data
# Only data with valid color and uncertainties in pm smaller than 0.4 and excluding foreground stars by color-cut
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
morralla ='/Users/amartinez/Desktop/morralla/'
cls_to_throw = '/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/'
pruebas = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
gns_clus = np.loadtxt('/Users/amartinez/Desktop/morralla/GNS1_f10c2_dmu1_at_GNS2_f4c3_2024-03-04 18:45:25.529303/cluster_num0_0_knn15_area3.00/cluster0_0_0_knn_15_area_3.00_pm_color.txt')

# 

ref_frame = 'ecuatorial'#TODO
# ref_frame = 'galactic'#TODO

# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>
choosen_cluster = 'Quintuplet'

cluster_gone = 'no'
center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet

if choosen_cluster == 'Arches':
    arches = ascii.read(catal + 'Arches_from_Article.txt') 
if choosen_cluster == 'Quintuplet':
    arches = ascii.read(catal + 'Quintuplet_from_Article.txt') 
# %%
t_inter = Time(['2022-05-27T00:00:00','2011-02-15T00:00:00'],scale='utc')
d_time = (t_inter[0]-t_inter[1]).to(u.yr)

# lets move Hosek stars to GNS2 epoch (2022) before the matching
# arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time.value/1000
# arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time.value/1000
arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time
arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time

center = (arches['F127M'] - arches['F153M'] >   1.7)
arches = arches[center]

hp = arches['Pclust']>0.1
arches = arches[hp]
RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])

arches.add_column(RA_DEC.ra, name='RA', index=-1)
arches.add_column(RA_DEC.dec, name='DEC', index=-1)

# sys.exit()
# lim_radec = np.where((RA_DEC.ra.value<max(gns_clus[:,0]))&(RA_DEC.ra.value>min(gns_clus[:,0]))
#                   &(RA_DEC.dec.value<max(gns_clus[:,1]))&(RA_DEC.dec.value>min(gns_clus[:,1])))

# RA_DEC = RA_DEC[lim_radec]
clus_cen = SkyCoord(ra = [np.mean(gns_clus[:,0])], dec = [np.mean(gns_clus[:,1])], unit = 'degree')
around_gns =  ap_coor.search_around_sky(clus_cen,RA_DEC,40*u.arcsec)

arches = arches[around_gns[1]]
RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])


# %
around_ls = np.empty(0)
ind_gns = np.empty(0)
rad = .15*u.arcsec#!!!
for i in range(len(gns_clus)):
# for i in range(1):
    m_point = SkyCoord(ra = [gns_clus[i,0]], dec = [gns_clus[i,1]],
                       unit = 'degree')
    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,RA_DEC,rad)
    # around_ls.append(group_md)
    around_ls = np.append(around_ls,group_md)
    if len(group_md) >0:
        ind_gns = np.append(ind_gns,i).astype(int)
        
    # print(len(group_md))
# %
# import pandas as pd
ar_uni = np.unique(around_ls).astype(int)

fig, ax = plt.subplots(1,2)
ax[0].scatter(RA_DEC.ra[ar_uni], RA_DEC.dec[ar_uni])
ax[0].scatter(gns_clus[:,0][ind_gns], gns_clus[:,1][ind_gns])

# ax.hist2d(RA, DEC, bins = 100)

arches = arches[ar_uni]
RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])

RA = RA_DEC.ra
DEC = RA_DEC.dec

lc1, lc2, l1 , l2 = compare_lists(np.array([RA.value,DEC.value]).T,gns_clus[ind_gns],0,1,0,1,0.3)

color = ['r','g','k','y','b']# just checking the order
# for i in range(5):
#     ax[1].scatter(arches['RA'][l1][i], arches['DEC'][l1][i],color = color[i])
#     ax[1].scatter(gns_clus[ind_gns][:,0][l2][i], gns_clus[ind_gns][:,1][l2][i],color = color[i])

dpm_ra = arches['pmRA'][l1]-gns_clus[ind_gns][:,2][l2]
dpm_dec = arches['pmDE'][l2]-gns_clus[ind_gns][:,3][l2]
ax[1].hist(dpm_ra, histtype = 'step', label = '$\sigma \mu_{ra} = %.2f$'%(np.std(dpm_ra)))
ax[1].hist(dpm_dec, histtype = 'step', label = '$\sigma \mu_{dec} = %.2f$'%(np.std(dpm_dec)))
ax[1].legend()

# sys.exit()
# m,(_,_)= aa.find_transform(gns_clus[:,0:2],np.array([RA.value,DEC.value]).T,max_control_points=300)
# m,(_,_)= aa.find_transform(gns_clus[:,0:2][ind_gns],np.array([RA.value,DEC.value]).T,max_control_points=50)
m, (s_ls, t_ls)= aa.find_transform(gns_clus[:,0:2][ind_gns],np.array([RA.value,DEC.value]).T,max_control_points=50)

print("Translation: (x, y) = (%.2f, %.2f)"%(m.translation[0],m.translation[1]))
print("Rotation: %.1f degrees"%(m.rotation * 180.0 /np.pi)) 
# print("Rotation: %s"%(m.rotation))
print("Scale factor: %.4f"%(m.scale))


gns_copy = copy.deepcopy(gns_clus[:,0:2][ind_gns])
gns_aa = aa.matrix_transform(gns_copy, m)
# gns_clus[:,0:2] = aa.matrix_transform(gns_clus[:,0:2], m)


fig, ax = plt.subplots(1,2)
ax[0].scatter(RA, DEC)
# ax.scatter(gns_clus[:,0], gns_clus[:,1])
ax[1].scatter(gns_aa[:,0],gns_aa[:,1], color = '#ff7f0e')
fig, ax = plt.subplots(1,1)
ax.scatter(RA, DEC)
# ax.scatter(gns_clus[:,0], gns_clus[:,1])
ax.scatter(gns_aa[:,0],gns_aa[:,1])

# for i in range(len(color)):
#     ax.scatter(s_ls[i][0], s_ls[i][1], color =color[i])
# # ax.scatter(gns_clus[:,0], gns_clus[:,1])
#     ax.scatter(t_ls[i][0],t_ls[i][1], color = color[i])
# %%

g_pmra = []
h_pmra = []
g_pmdec = []
h_pmdec = []
for i in range(len(s_ls)):
    g_aa = (gns_clus[ind_gns][:,0] == s_ls[i][0]) & (gns_clus[ind_gns][:,1] == s_ls[i][1])
    g_pmra.append(gns_clus[ind_gns][g_aa][0][2])
    g_pmdec.append(gns_clus[ind_gns][g_aa][0][3])
    
    h_aa = (arches['RA'] == t_ls[i][0]) & (arches['DEC'] == t_ls[i][1])
    h_pmra.append(arches['pmRA'][h_aa].value)
    h_pmdec.append(arches['pmDE'][h_aa].value)
    # print(gns_clus[ind_gns][g_aa][0][3])
    # print(arches['pmRA'][h_aa].value)

df_pmra = np.array(h_pmra).flatten()-(np.array(g_pmra))
df_pmdec = np.array(h_pmdec).flatten()-(np.array(g_pmdec))

fig, ax = plt.subplots(1,1)
ax.hist(df_pmra,histtype ='step', label = '$\sigma \mu_{ra} = %.2f$'%(np.std(df_pmra)))
ax.hist(df_pmdec,histtype ='step', label = '$\sigma \mu_{dec} = %.2f$'%(np.std(df_pmdec)))
ax.legend()    


