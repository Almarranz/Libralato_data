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
from astropy.table import Table
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
pruebas = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas'
pruebas_hosek = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'



gns_all = np.loadtxt('/Users/amartinez/Desktop/morralla/GNS1_f10c2_dmu1_at_GNS2_f4c3_2024-03-15 10:49:03.111142/cluster_num0_0_knn15_area3.00/cluster0_0_0_knn_50_area_2.84_pm_color.txt')
gns_clus = np.loadtxt('/Users/amartinez/Desktop/morralla/GNS1_f10c2_dmu1_at_GNS2_f4c3_2024-03-15 10:49:03.111142/cluster_num0_0_knn15_area3.00/cluster0_0_0_knn_50_area_2.84_pm_color.txt')

gns_clus = gns_clus[gns_clus[:,-1].argsort()]
perc_b_gns = 0.5
bright_gns = int(len(gns_clus)*perc_b_gns)
gns_clus = gns_clus[0: bright_gns]
print(len(gns_clus))
# =============================================================================
# gns_clus = np.loadtxt('/Users/amartinez/Desktop/PhD/HAWK/pm_gns1_gns2_off/pm_GaiaRF_ep1_f10c2_ep2_f4c3deg2_dmax100_sxy2.txt')
# gns_clus[:,0] = gns_clus[:,14]
# gns_clus[:,1] = gns_clus[:,15]
# =============================================================================
ref_frame = 'ecuatorial'#TODO
# ref_frame = 'galactic'#TODO

# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>
choosen_cluster = 'Quintuplet'

cluster_gone = 'no'
if choosen_cluster == 'Arches':
    center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') 
    arches = ascii.read(catal + 'Arches_from_Article.txt') 
elif choosen_cluster == 'Quintuplet':
    center_arc = SkyCoord('17h46m14.68579s', '-28d49m38.99169s', frame='icrs',obstime ='J2016.0',equinox = 'J2000')#Quintuplet
    arches = ascii.read(catal + 'Quintuplet_from_Article.txt') 
    arches_all = ascii.read(catal + 'Quintuplet_from_Article.txt') 

perc_b = 0.3
bright = int(len(arches)*perc_b)
arches = arches[0:bright]



# Select the time you have to move the stars according to the observation time
# t_inter = Time(['2022-05-27T00:00:00','2011-02-15T00:00:00'],scale='utc')
t_inter = Time(['2015-06-08T00:00:00','2011-06-15T00:00:00'],scale='utc')
d_time = (t_inter[0]-t_inter[1]).to(u.yr)

# lets move Hosek stars to GNS2 epoch (2022) before the matching
arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time.value/1000
arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time.value/1000
# arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time
# arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time

RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])

RA = RA_DEC.ra
DEC = RA_DEC.dec
arches.add_column(RA_DEC.ra, name='RA', index=-1)
arches.add_column(RA_DEC.dec, name='DEC', index=-1)
center = (arches['F127M'] - arches['F153M'] >   1.7)
# arches = arches[center]

epm_l = 0.5
epm = (arches['e_pmRA']<epm_l) & (arches['e_pmDE']<epm_l)
arches = arches[epm]
RA_DEC = RA_DEC[epm]

def make_coord(ra,dec):
    gns_coord = SkyCoord(ra = ra, dec = dec, unit = 'degree',
                         frame = 'fk5',equinox ='J2000',obstime='J2015.5')
    return gns_coord

# gns_coord = SkyCoord(ra = gns_clus[:,0], dec = gns_clus[:,1], unit = 'degree',
#                      frame = 'fk5',equinox ='J2000',obstime='J2015.5')
gns_coord = make_coord(gns_clus[:,0], gns_clus[:,1])

gns_coord = gns_coord.transform_to('icrs')
gns_dra, gns_ddec = gns_coord.spherical_offsets_to(center_arc.frame)

gns_dra = gns_dra.to(u.arcsec)
gns_ddec = gns_ddec.to(u.arcsec)
gns_off_coor = SkyCoord(ra = gns_dra, dec = gns_ddec)

gns_clus = np.c_[gns_clus, gns_dra.value*-1+0.08, gns_ddec.value*-1-0.01]

r_aro = 40*u.arcsec
gns_cent = SkyCoord(ra = [np.mean(gns_dra)] , dec = [np.mean(gns_ddec)])
RA_DEC_off = SkyCoord(ra = arches['dRA'], dec = arches['dDE'])
around_arc =  ap_coor.search_around_sky(gns_cent,RA_DEC_off,r_aro)
around_gns =  ap_coor.search_around_sky(gns_cent,gns_off_coor,r_aro)

arches = arches[around_arc[1]]
RA_DEC = RA_DEC[around_arc[1]]
gns_clus = gns_clus[around_gns[1]]

gns_coord = make_coord(gns_clus[:,0], gns_clus[:,1])


# %%
idx = RA_DEC.match_to_catalog_sky(gns_coord)
valid = np.where(idx[1]<0.1*u.arcsec)#!!!
arches_match=arches[valid]
gns_match=gns_clus[idx[0][valid]]


dmu_ra = arches_match['pmRA']- gns_match[:,2]
dmu_de = arches_match['pmDE']- gns_match[:,3]

print('mu_ra = %.2f +- %.2f'%(np.mean(dmu_ra),np.std(dmu_ra)))
print('mu_de = %.2f +- %.2f'%(np.mean(dmu_de),np.std(dmu_de)))
print('Mathing stars =',len(arches_match))
# print(np.mean(dmu_ra))
# print(np.std(dmu_ra))
# print(np.mean(dmu_de))
# print(np.std(dmu_de))
# %%
fig, ax = plt.subplots(1,3)

# ax.scatter(arches_all['dRA'], arches_all['dDE'], label ='Hosek all')
ax[0].scatter(gns_clus[:,-2], gns_clus[:,-1], label = 'GNS offsets')
ax[0].scatter(arches['dRA'], arches['dDE'], label ='Hosek (brighter %.1f %%)'%(perc_b*100),zorder = 0)
# ax.scatter(arc_cen_off_ra,arc_cen_off_dec)
ax[0].legend(fontsize = 10)
ax[0].axis('scaled')

ax[1].scatter(arches_all['F153M'],arches_all['e_pmRA'], color = 'k', s =5, alpha = 0.1)
ax[1].scatter(arches['F153M'],arches['e_pmRA'], color =  '#ff7f0e')
ax[1].scatter(arches_match['F153M'],arches_match['e_pmDE'], color = 'green')

ax[2].hist(arches_match['F153M']-gns_match[:,-3])

# sys.exit()
# %%

hos_off_b= np.array([arches_match['dRA'].value,arches_match['dDE'].value]).T
gns_off_b = np.array([gns_match[:,-2], gns_match[:,-1]]).T
tb,(sr_ls,tr_ls) =aa.find_transform(hos_off_b,gns_off_b,max_control_points=len(arches_match))

print("Translation: (x, y) = (%.2f, %.2f)"%(tb.translation[0],tb.translation[1]))
print("Rotation: %.1f degrees"%(tb.rotation * 180.0 /np.pi)) 
print("Scale factor: %.4f"%(tb.scale))
print('len(sr_ls)',len(sr_ls))


# gns_off_b1 = aa.matrix_transform(gns_off_b, tb)
# tb1,(sr_ls1,tr_ls1) =aa.find_transform(hos_off_b,gns_off_b1,max_control_points=40)













# =============================================================================
# hp = arches['Pclust']>0.1
# arches = arches[hp]
# RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
# 
# arches.add_column(RA_DEC.ra, name='RA', index=-1)
# arches.add_column(RA_DEC.dec, name='DEC', index=-1)
# 
# # sys.exit()
# # lim_radec = np.where((RA_DEC.ra.value<max(gns_clus[:,0]))&(RA_DEC.ra.value>min(gns_clus[:,0]))
# #                   &(RA_DEC.dec.value<max(gns_clus[:,1]))&(RA_DEC.dec.value>min(gns_clus[:,1])))
# 
# # RA_DEC = RA_DEC[lim_radec]
# clus_cen = SkyCoord(ra = [np.mean(gns_clus[:,0])], dec = [np.mean(gns_clus[:,1])], unit = 'degree')
# around_gns =  ap_coor.search_around_sky(clus_cen,RA_DEC,40*u.arcsec)
# # 
# arches = arches[around_gns[1]]
# RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
# 
# 
# # %
# around_ls = np.empty(0)
# ind_gns = np.empty(0)
# rad = .15*u.arcsec#!!!
# for i in range(len(gns_clus)):
# # for i in range(1):
#     m_point = SkyCoord(ra = [gns_clus[i,0]], dec = [gns_clus[i,1]],
#                        unit = 'degree')
#     idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,RA_DEC,rad)
#     # around_ls.append(group_md)
#     around_ls = np.append(around_ls,group_md)
#     if len(group_md) >0:
#         ind_gns = np.append(ind_gns,i).astype(int)
#         
#     # print(len(group_md))
# # %
# # import pandas as pd
# ar_uni = np.unique(around_ls).astype(int)
# 
# fig, ax = plt.subplots(1,2)
# ax[0].scatter(RA_DEC.ra[ar_uni], RA_DEC.dec[ar_uni])
# ax[0].scatter(gns_clus[:,0][ind_gns], gns_clus[:,1][ind_gns])
# 
# # ax.hist2d(RA, DEC, bins = 100)
# 
# arches = arches[ar_uni]
# RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
# 
# RA = RA_DEC.ra
# DEC = RA_DEC.dec
# 
# lc1, lc2, l1 , l2 = compare_lists(np.array([RA.value,DEC.value]).T,gns_clus[ind_gns],0,1,0,1,0.3)
# 
# color = ['r','g','k','y','b']# just checking the order
# # for i in range(5):
# #     ax[1].scatter(arches['RA'][l1][i], arches['DEC'][l1][i],color = color[i])
# #     ax[1].scatter(gns_clus[ind_gns][:,0][l2][i], gns_clus[ind_gns][:,1][l2][i],color = color[i])
# 
# dpm_ra = arches['pmRA'][l1]-gns_clus[ind_gns][:,2][l2]
# dpm_dec = arches['pmDE'][l2]-gns_clus[ind_gns][:,3][l2]
# ax[1].hist(dpm_ra, histtype = 'step', label = '$\sigma \mu_{ra} = %.2f$'%(np.std(dpm_ra)))
# ax[1].hist(dpm_dec, histtype = 'step', label = '$\sigma \mu_{dec} = %.2f$'%(np.std(dpm_dec)))
# ax[1].legend()
# 
# # sys.exit()
# # m,(_,_)= aa.find_transform(gns_clus[:,0:2],np.array([RA.value,DEC.value]).T,max_control_points=300)
# # m,(_,_)= aa.find_transform(gns_clus[:,0:2][ind_gns],np.array([RA.value,DEC.value]).T,max_control_points=50)
# m, (s_ls, t_ls)= aa.find_transform(gns_clus[:,0:2][ind_gns],np.array([RA.value,DEC.value]).T,max_control_points=50)
# 
# print("Translation: (x, y) = (%.2f, %.2f)"%(m.translation[0],m.translation[1]))
# print("Rotation: %.1f degrees"%(m.rotation * 180.0 /np.pi)) 
# # print("Rotation: %s"%(m.rotation))
# print("Scale factor: %.4f"%(m.scale))
# 
# 
# gns_copy = copy.deepcopy(gns_clus[:,0:2][ind_gns])
# gns_aa = aa.matrix_transform(gns_copy, m)
# # gns_clus[:,0:2] = aa.matrix_transform(gns_clus[:,0:2], m)
# 
# 
# fig, ax = plt.subplots(1,2)
# ax[0].scatter(RA, DEC)
# # ax.scatter(gns_clus[:,0], gns_clus[:,1])
# ax[1].scatter(gns_aa[:,0],gns_aa[:,1], color = '#ff7f0e')
# fig, ax = plt.subplots(1,1)
# ax.scatter(RA, DEC)
# # ax.scatter(gns_clus[:,0], gns_clus[:,1])
# ax.scatter(gns_aa[:,0],gns_aa[:,1])
# 
# # for i in range(len(color)):
# #     ax.scatter(s_ls[i][0], s_ls[i][1], color =color[i])
# # # ax.scatter(gns_clus[:,0], gns_clus[:,1])
# #     ax.scatter(t_ls[i][0],t_ls[i][1], color = color[i])
# # %%
# 
# g_pmra = []
# h_pmra = []
# g_pmdec = []
# h_pmdec = []
# for i in range(len(s_ls)):
#     g_aa = (gns_clus[ind_gns][:,0] == s_ls[i][0]) & (gns_clus[ind_gns][:,1] == s_ls[i][1])
#     g_pmra.append(gns_clus[ind_gns][g_aa][0][2])
#     g_pmdec.append(gns_clus[ind_gns][g_aa][0][3])
#     
#     h_aa = (arches['RA'] == t_ls[i][0]) & (arches['DEC'] == t_ls[i][1])
#     h_pmra.append(arches['pmRA'][h_aa].value)
#     h_pmdec.append(arches['pmDE'][h_aa].value)
#     # print(gns_clus[ind_gns][g_aa][0][3])
#     # print(arches['pmRA'][h_aa].value)
# 
# df_pmra = np.array(h_pmra).flatten()-(np.array(g_pmra))
# df_pmdec = np.array(h_pmdec).flatten()-(np.array(g_pmdec))
# 
# fig, ax = plt.subplots(1,1)
# ax.hist(df_pmra,histtype ='step', label = '$\sigma \mu_{ra} = %.2f$'%(np.std(df_pmra)))
# ax.hist(df_pmdec,histtype ='step', label = '$\sigma \mu_{dec} = %.2f$'%(np.std(df_pmdec)))
# ax.legend()    
# 
# 
# =============================================================================
