#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:01:02 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.coordinates as ap_coor
import pandas as pd
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
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

# folder = '/Users/amartinez/Desktop/morralla/Sec_A_all_color_good_VvsL_mass/'
folder = '/Users/amartinez/Desktop/morralla/Sec_A_all_good_VvsL_mass/'

# %%
name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

# %%
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
section = 'A'#selecting the whole thing

if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%
# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

color_de_cluster = 'lime'

# for clus_f in glob.glob(folder +'*'):
# % coordinates
ra_=catal[:,5]
dec_=catal[:,6]
# Process needed for the trasnformation to galactic coordinates
coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
gal_c=coordenadas.galactic

t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))

mul,mub = catal[:,-6],catal[:,-5]
x,y = catal[:,7], catal[:,8]
colorines = catal[:,3]-catal[:,4]
# %%Thi is for the extinction

Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()#TODO
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
AKs_center =AKs_np[center]#TODO
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree)
# %
# %
AKs_list1 =  np.arange(1.6,2.11,0.01)
AKs_list = np.append(AKs_list1,0)#I added the 0 for the isochrones without extiction
# %%
for clus_f in glob.glob(folder +'cluster_num3_0_knn7_area2.12'):
    
    print(clus_f)
    all_clus = []
    cluster_len =[]
    for file in glob.glob(clus_f + '/cluster*'):
        # ra, dec, l, b, pml, pmb,J, H, Ks,x, y, AKs_mean, dAks_mean, radio("),cluster_ID
        print(file)
        cluster = np.loadtxt(file)
        cluster_len.append(len(cluster))
        for line in range(len(cluster)):
            all_clus.append(cluster[line])
    clus_arr = np.array(all_clus)
# %
    clus_arr[np.isnan(clus_arr)]=-99
    clus_trim = np.delete(clus_arr,-1,axis=1)#trim this to be able to compare both lists

    cluster_unique = np.unique(clus_trim,axis =0)# select only uniques stars
    if any(x<len(cluster_unique) for x in cluster_len):
        print('some stars added')
    else:
        print('all cluster are the same')

    fig, ax = plt.subplots(1,3,figsize=(30,10))
    ax[0].scatter(catal[:,-6],catal[:,-5],color = 'k', alpha = 0.1, zorder=1)
    ax[0].scatter(cluster_unique[:,4],cluster_unique[:,5], color = color_de_cluster, zorder=3,s=100) 
    ax[0].set_xlim(-10,10)
    ax[0].set_ylim(-10,10)
    ax[0].invert_xaxis()
    
    mul_sig, mub_sig = np.std(cluster_unique[:,4]), np.std(cluster_unique[:,5])
    mul_mean, mub_mean = np.mean(cluster_unique[:,4]), np.mean(cluster_unique[:,5])
    
    mul_sig_all, mub_sig_all = np.std(catal[:,-6]), np.std(catal[:,-5])
    mul_mean_all, mub_mean_all = np.mean(catal[:,-6]), np.mean(catal[:,-5])
    
    
    vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
    vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
    
    propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
    propiedades_all = dict(boxstyle='round', facecolor='k', alpha=0.1)
    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
        verticalalignment='top', bbox=propiedades)
    ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
        verticalalignment='top', bbox=propiedades_all)
    
    
    c2 = SkyCoord(ra = cluster_unique[:,0]*u.deg,dec = cluster_unique[:,1]*u.deg)
    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
    rad = max(sep)/2
    
    m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)])
    
    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*3)
    
    ax[0].scatter(catal[:,-6][group_md],catal[:,-5][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)

    prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
    ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(cluster_unique)), transform=ax[1].transAxes, fontsize=30,
                            verticalalignment='top', bbox=prop)
    
    ax[1].scatter(catal[:,7][group_md],catal[:,8][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.3)

    ax[1].scatter(catal[:,7],catal[:,8],color ='k',alpha = 0.1)
    ax[1].scatter(cluster_unique[:,9], cluster_unique[:,10], color = color_de_cluster,s=100)
    
    
    ax[2].scatter(catal[:,3]-catal[:,4],catal[:,4], color='k' ,s=50,zorder=1, alpha=0.03)
    ax[2].scatter(cluster_unique[:,7]-cluster_unique[:,8],cluster_unique[:,8], color=color_de_cluster ,s=120,zorder=3, alpha=1)
    ax[2].invert_yaxis()  
    ax[2].set_xlim(1.3,3)
    ax[2].set_xlabel('$H-Ks$',fontsize =30)
    ax[2].set_ylabel('$Ks$',fontsize =30)
    
    # ax[2].set_title('Cluster %s, eps = %s'%(clus_num,round(eps_av,3)))
    txt_around = '\n'.join(('H-Ks =%.3f'%(cluster_unique[:,7]-cluster_unique[:,8]),
                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(cluster_unique[:,7]-cluster_unique[:,8])),
                         'diff_color = %.3f'%(max(cluster_unique[:,7]-cluster_unique[:,8])-min(cluster_unique[:,7]-cluster_unique[:,8]))))
    props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
    ax[2].text(0.50, 0.25,txt_around, transform=ax[2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props_arou)




