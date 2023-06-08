#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:09:59 2023

@author: amartinez
"""

# Throwing cluster to Libralato and check detention threshold with dbscan
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
import numpy as np
from matplotlib import rcParams
from kneed import DataGenerator, KneeLocator
import pandas as pd
from astropy import units as u
from astropy.table import QTable
import pylab as p
from random import seed
from random import random
import glob
import os
import math
import shutil
from datetime import datetime
from tabulate import tabulate
from texttable import Texttable
import latextable
from extinction_simulation import extinction_sim
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
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
name='WFC3IR'
morralla ='/Users/amartinez/Desktop/morralla/'
cls_to_throw = '/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/'

trimmed_data='no'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

section = 'B'#selecting the whole thing
ref_frame ='ecuatorial'#TODO
MS_ra,MS_dec = np.loadtxt(cata + 'MS_section%s.txt'%(section),unpack=True, usecols=(0,1),skiprows=0)
MS_coord = SkyCoord(ra = MS_ra*u.deg, dec = MS_dec*u.deg, frame = 'icrs',equinox ='J2014.2')
if section == 'All':
    # RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))




# radio = 2
age_ff = 0
samples_dist = 35
clustered_by = 'all_color'#TODO
# clustered_by = 'all'#TODO1
# clustered_by = 'vel_col'#TODO
# clustered_by = 'vel'
# 'RA, DEC, pmra, pmdec, f127, f153,'
clus_now = np.loadtxt(cls_to_throw + 'Arches_0.00evol_times.txt')
core_cluster = np.loadtxt(cls_to_throw + 'Arches_%.2fevol_times.txt'%(age_ff))

c2 = SkyCoord(ra = core_cluster[:,0],dec =core_cluster[:,1], unit ='degree',  equinox = 'J2000', obstime = 'J2015.4')
sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
rad = max(sep)/2
with open(pruebas + 'stad_arches_nn%s_clustby_(%s)_%.2fevol_times.txt'%(samples_dist,clustered_by,age_ff), 'w') as file:
    file.write('# pm_ra pm_dec sigma_{pmra} sigma_{pmdec} radio #stars %% of the origibal custrer found (first line: data of the original thrown cluster)\n%.4f %.4f %.4f %.4f %.2f %s\n'%(np.mean(core_cluster[:,2]),np.mean(core_cluster[:,3]),
                                      np.std(core_cluster[:,2]),np.std(core_cluster[:,3]),rad.to(u.arcsec).value, len(core_cluster)))
    file.close()
for loop in range(3):# This is the number of times that the cluster is thrwon into the data
   
    ran_id = int(np.random.uniform(0,len(catal),1))
    
    
    ra_rand =  catal[ran_id,0]
    dec_rand = catal[ran_id,1]
    ra_cen, dec_cen  = np.mean(core_cluster[:,0]), np.mean(core_cluster[:,1])
    
    core_cluster[:,0] -= ra_cen
    core_cluster[:,1] -= dec_cen
     
    core_cluster[:,0] += ra_rand
    core_cluster[:,1] += dec_rand
    
    # Give the cluster the values for H and Ks according with the extiction of the area where it has been thrown
    core_cluster = extinction_sim(clus_now, core_cluster)
        
   
    # %%
    # We insert the core_cluster into the Libralato data by creating an array of 
    # zeros with the same shape and filling it with the values of the core_cluster
    # RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua'9,'dmua'10,'mud'11,'dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
    cluster_insert = np.zeros((len(core_cluster),23))
    cluster_insert[:,0:2] = core_cluster[:,0:2]
    cluster_insert[:,3:5] = core_cluster[:,-2:]
    cluster_insert[:,9],cluster_insert[:,11] = core_cluster[:,2],core_cluster[:,3]
    
    # Add the cluster to libralato data
    catal_w = np.r_[catal,cluster_insert]
    no_clus = np.full(len(catal),0)
    si_clus = np.full(len(cluster_insert),1)
    indx = np.r_[no_clus,si_clus]
    catal_w = np.c_[catal_w,indx]
    
    
    
    
    
    # %%
    
    # catal_w = catal
    valid=np.where(np.isnan(catal_w[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal_w=catal_w[valid]
   
    
    # Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
    center_definition='G_G'#this variable can be L_G or G_G
    if center_definition =='L_G':
        valid=np.where(np.isnan(catal_w[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
        catal_w=catal_w[valid]
        center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
    elif center_definition =='G_G':
        valid=np.where((np.isnan(catal_w[:,3])==False) | (np.isnan(catal_w[:,4])==False ))
        catal_w=catal_w[valid]
        center=np.where(catal_w[:,3]-catal_w[:,4]>1.3)
    catal_w=catal_w[center]
    
    
    
    
    RA = catal_w[:,0]
    DEC = catal_w[:,1]
    pmra, pmdec= catal_w[:,9], catal_w[:,11]
    colorines = catal_w[:,3] - catal_w[:,4]
    
    pmra_kernel, pmdec_kernel = gaussian_kde(catal_w[:,9]), gaussian_kde(catal_w[:,11])
    raoff_kernel, decoff_kernel = gaussian_kde(RA), gaussian_kde(DEC)
    color_kernel = gaussian_kde(colorines)
    
    if clustered_by == 'all_color':
        # X = np.array([pmra,pmdec,raoff,decoff,colorines]).T
        X = np.array([pmra,pmdec,RA,DEC,colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    elif clustered_by == 'all':
        X = np.array([pmra,pmdec,RA,DEC]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    elif clustered_by == 'vel_col':
        X = np.array([pmra,pmdec,colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    elif clustered_by == 'vel':
        X = np.array([pmra,pmdec]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    
    
    lst_d_KNN_sim = []
    for d in range(2):
        mudec_sim,  mura_sim = pmdec_kernel.resample(len(pmdec)), pmra_kernel.resample(len(pmra))
        raoff_sim, decoff_sim = raoff_kernel.resample(len(pmdec)), decoff_kernel.resample(len(pmdec))
        color_sim = color_kernel.resample(len(pmdec))
        if clustered_by == 'all_color':
            X_sim=np.array([mura_sim[0],mudec_sim[0],raoff_sim[0],decoff_sim[0],color_sim[0]]).T
            X_stad_sim = StandardScaler().fit_transform(X_sim)
            tree_sim =  KDTree(X_stad_sim, leaf_size=2)
            
            dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
            
            lst_d_KNN_sim.append(min(d_KNN_sim))
        elif clustered_by =='all':
            X_sim=np.array([mura_sim[0],mudec_sim[0],raoff_sim[0],decoff_sim[0]]).T
            X_stad_sim = StandardScaler().fit_transform(X_sim)
            tree_sim =  KDTree(X_stad_sim, leaf_size=2)
            
            dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
            
            lst_d_KNN_sim.append(min(d_KNN_sim))
        elif clustered_by =='vel_col':
            X_sim=np.array([mura_sim[0],mudec_sim[0],color_sim[0]]).T
            X_stad_sim = StandardScaler().fit_transform(X_sim)
            tree_sim =  KDTree(X_stad_sim, leaf_size=2)
            
            dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
            
            lst_d_KNN_sim.append(min(d_KNN_sim))
            
        elif clustered_by =='vel':
            X_sim=np.array([mura_sim[0],mudec_sim[0]]).T
            X_stad_sim = StandardScaler().fit_transform(X_sim)
            tree_sim =  KDTree(X_stad_sim, leaf_size=2)
            
            dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
            
            lst_d_KNN_sim.append(min(d_KNN_sim))
    
    d_KNN_sim_av = np.mean(lst_d_KNN_sim)
    # %%
# =============================================================================
#     # Plotting knn histogram
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     # ax.set_title('Number of points = %s '%(len(pmdec)))
#     
#     ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k', linewidth = 5,label = 'With cluster (min. = %.2f)'%(min(d_KNN)))
#     ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', linewidth = 5,alpha = 0.5,label = 'Simulated (min. = %.2f)'%(d_KNN_sim_av))
#     ax.set_xlabel('%s-NN distance'%(samples_dist)) 
#     ax.set_xlim(0,1)
#     ax.set_ylabel('N') 
#     ax.legend()
# =============================================================================
    
    eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
    texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                        'min sim d_KNN =%s'%(round(d_KNN_sim_av,3)),'average = %s'%(eps_av)))
    
   
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    # place a text box in upper left in axes coords
    # ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
    #     verticalalignment='top', bbox=props)
    
    
    
    # %%
    # =============================================================================
    # DBSCAN part
    # =============================================================================
    
    clustering = DBSCAN(eps = eps_av, min_samples=samples_dist).fit(X_stad)
    
    l_c=clustering.labels_
    
    n_clusters = len(set(l_c)) - (1 if -1 in l_c else 0)
    n_noise=list(l_c).count(-1)
    
    u_labels = set(l_c)
    colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l_c)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
    # colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
    #  '#7f7f7f', '#bcbd22', '#17becf']
    # colors =[(0,1,0,1)]
    
    for k in range(len(colors)): #give noise color black with opacity 0.1
        if list(u_labels)[k] == -1:
            colors[k]=[0,0,0,0.1]
            
    colores_index=[]      
    for c in u_labels:
        cl_color=np.where(l_c==c)
        colores_index.append(cl_color)
        
    # %
    # default colors = '#1f77b4', '#ff7f0e'
    elements_in_cluster=[]
    # Make the stadistics without plotting anything (faster)
# =============================================================================
#     for i in range(len(set(l_c))-1):
#         mura_mean, mudec_mean = np.mean(pmra[colores_index[i]]), np.mean(pmdec[colores_index[i]])
#         mura_sig,  mudec_sig = np.std(pmra[colores_index[i]]), np.std(pmdec[colores_index[i]])
#         c2 = SkyCoord(ra = RA[colores_index[i]],dec =DEC[colores_index[i]], unit ='degree',  equinox = 'J2000', obstime = 'J2015.4')
#         sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
#         rad = max(sep)/2
#         ori_clus = np.where(catal_w[:,-1][colores_index[i]] == 1)
#         K_ts = catal_w[:,4][colores_index[i]]
#         print(len('loop numer %s')*'_'+'\n'+'loop numer %s'%(loop)+'\n'+len('loop numer %s')*'_')
#         if len(ori_clus[0])>0:
#             with open(pruebas + 'stad_arches_nn%s_clustby_(%s)_%.2fevol_times.txt'%(samples_dist,clustered_by,age_ff), 'a') as file:
#                 file.write('%.4f %.4f %.4f %.4f %.2f %s %.0f\n'%(mura_mean,mudec_mean,mura_sig,mudec_sig,rad.to(u.arcsec).value,len(colores_index[i][0]),len(ori_clus[0])/len(core_cluster)*100))
# =============================================================================
    # Makes the plotting
    for i in range(len(set(l_c))-1):
    # for i in range(1):
        fig, ax = plt.subplots(1,3,figsize=(30,10))
        ax[0].set_title('%s evol times (%.2f Myr)'%(age_ff, age_ff*1.5))
        # ax[0].set_xticks(np.arange(-12,11))
        # ax[0].grid()
        ax[0].invert_xaxis()
        # ax[1].invert_xaxis()
        # ax[2].invert_yaxis()
        ax[1].set_xlim(min(catal[:,0]),max(catal[:,0]))
        ax[1].set_ylim(min(catal[:,1]),max(catal[:,1]))
        c2 = SkyCoord(ra = RA[colores_index[i]],dec =DEC[colores_index[i]], unit ='degree',  equinox = 'J2000', obstime = 'J2015.4')
        sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
        rad = max(sep)/2
        # ax[0].scatter(pmra[colores_index[i]], pmdec[colores_index[i]],color=colors[i],zorder=3)
        ax[0].scatter(pmra[colores_index[i]], pmdec[colores_index[i]],color = '#ff7f0e',zorder=3)
    
        # ax[1].scatter(l[colores_index[i]], b[colores_index[i]],color=colors[i],zorder=3)
        # ax[1].scatter(RA[colores_index[i]],DEC[colores_index[i]],color=colors[i],zorder=3,s=100,alpha =0.3)
        ax[1].scatter(RA[colores_index[i]],DEC[colores_index[i]],color = '#ff7f0e',s=100,alpha =1,zorder =2)
        
        # ax[1].scatter(gns_match[colores_index[i]][:,0],gns_match[colores_index[i]][:,2],color=colors[i],zorder=3,s=100)
        # ax[2].scatter(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]],arches['F153M'][colores_index[i]],color=colors[i],zorder=13)
        # ax[2].scatter(catal_w[:,3][colores_index[i]]-catal_w[:,4][colores_index[i]],catal_w[:,4][colores_index[i]],color = '#ff7f0e',zorder=3)
    
        mura_mean, mudec_mean = np.mean(pmra[colores_index[i]]), np.mean(pmdec[colores_index[i]])
        mura_sig,  mudec_sig = np.std(pmra[colores_index[i]]), np.std(pmdec[colores_index[i]])
        if ref_frame =='ecuatorial':
            vel_txt = '\n'.join(('$\mu_{ra}$ = %s,$\mu_{dec}$ = %s'%(round(mura_mean,3), round(mudec_mean,3)),
                                 '$\sigma_{\mu ra}$ = %s, $\sigma_{\mu dec}$ = %s'%(round(mura_sig,3), round(mudec_sig,3))))   
        if ref_frame =='galactic':
            vel_txt = '\n'.join(('$\mu_{l}$ = %s,$\mu_{b}$ = %s'%(round(mura_mean,3), round(mudec_mean,3)),
                                 '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mura_sig,3), round(mudec_sig,3))))   
        # propiedades = dict(boxstyle='round', facecolor=colors[i] , alpha=0.2)
        propiedades = dict(boxstyle='round', facecolor= '#ff7f0e', alpha=0.2)
    
        ax[0].text(0.15, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
            verticalalignment='top', bbox=propiedades)
        # prop = dict(boxstyle='round', facecolor=colors[i] , alpha=0.2)
        prop = dict(boxstyle='round', facecolor='#ff7f0e' , alpha=0.2)
    
        ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
                                verticalalignment='top', bbox=prop)
        txt_color = '\n'.join(('H-Ks =%.3f'%(np.median(catal_w[:,3][colores_index[i]]-catal_w[:,4][colores_index[i]])),
                                                '$\sigma_{H-Ks}$ = %.3f'%(np.std(catal_w[:,3][colores_index[i]]-catal_w[:,4][colores_index[i]]))))
                                                # 'diff_color = %.3f'%(max(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]])-min(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]]))))
        # props = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
        props = dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.2)
    
        # ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
                                # verticalalignment='top', bbox=props)
        ax[0].scatter(pmra[colores_index[-1]], pmdec[colores_index[-1]],color=colors[-1],zorder=1, alpha=0.01)
        
        ax[0].set_xlabel(r'$\mathrm{\mu_{ra*} (mas\ yr^{-1})}$',fontsize =30) 
        ax[0].set_ylabel(r'$\mathrm{\mu_{dec} (mas\ yr^{-1})}$',fontsize =30) 
        # ax[1].scatter(l[colores_index[-1]], b[colores_index[-1]],color=colors[-1],zorder=1)
        ax[1].scatter(RA[colores_index[-1]],DEC[colores_index[-1]],color=colors[-1],s=5,alpha = 0.01,zorder = 1)
        # ax[1].scatter(RA,DEC,color=colors[-1],zorder=3,s=100,alpha = 0.01)
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax[1].invert_xaxis()
        ax[1].scatter(core_cluster[:,0], core_cluster[:,1], color = 'k', alpha = 0.5,zorder =3) 
        ori_clus = np.where(catal_w[:,-1][colores_index[i]] == 1)
        if len(ori_clus[0]) > 0:
            ax[1].scatter(catal_w[:,0][colores_index[i]][ori_clus],catal_w[:,1][colores_index[i]][ori_clus],color = 'green',s=20,zorder =3, label = 'Original Recovered stars = %s (%.0f %%) \n out of %s'%(len(ori_clus[0]),len(ori_clus[0])/len(core_cluster)*100,len(core_cluster)))
            ax[1].legend(loc = 4)
            with open(pruebas + 'stad_arches_nn%s_clustby_(%s)_%.2fevol_times.txt'%(samples_dist,clustered_by,age_ff), 'a') as file:
                file.write('%.4f %.4f %.4f %.4f %.2f %s %.0f\n'%(mura_mean,mudec_mean,mura_sig,mudec_sig,rad.to(u.arcsec).value,len(colores_index[i][0]),len(ori_clus[0])/len(core_cluster)*100))
                
        # ax[1].set_yticks([-29.025, -29.   , -28.975, -28.95, -28.9  , -28.875,
        #         -28.85 , -28.825])
        # nt = np.array([list(ax[1].get_yticks()).remove(-28.925)])
        # ax[1].set_yticks(nt)
        ax[1].invert_xaxis()
        ax[1].set_xlabel('ra(deg)',fontsize =30) 
        ax[1].set_ylabel('dec(deg)',fontsize =30)
        # ax[1].yaxis.set_label_coords(-0.05, 0.5)
        
        ax[2].scatter(catal_w[:,3][colores_index[-1]]-catal_w[:,4][colores_index[-1]],catal_w[:,4][colores_index[-1]],color=colors[-1],zorder=1, s = 1, alpha = 0.1)
        ax[2].scatter(catal_w[:,3][colores_index[i]]-catal_w[:,4][colores_index[i]],catal_w[:,4][colores_index[i]],color = '#ff7f0e',s= 50)

        ax[2].invert_yaxis()
        ax[2].set_xlabel('H-Ks',fontsize =30) 
        ax[2].set_ylabel('Ks',fontsize =30) 
    
        # ax[0].scatter(pmra, pmdec,color = 'k')
        ax[1].scatter(catal_w[:,0], catal_w[:,1])
       
        plt.show()

        
# %%
# =============================================================================
# # Analyzing the statistics
# 
# # pm_ra pm_dec sigma_{pmra} sigma_{pmdec} radio #stars 
# data = np.loadtxt(pruebas + 'stad_arches_nn%s_clustby_(%s)_%.2fMyr.txt'%(samples_dist,clustered_by,age_ff))    
# 
# model = data[0]
# sta = data[1:]
# 
# # %%
# rows = [['Cluster','$\mu_{ra}$','$\mu_{dec}$','$\sigma_{\mu dec}$','$\sigma_{\mu dec}$', 'Radio','stars'],
#         ['Model',str('%.2f'%(model[0])),'%.2f'%(model[1]),'%.2f'%(model[2]),'%.2f'%(model[3]),'%.2f'%(model[4]),'%.2f'%(model[5])],
#         ['Recovered','%.2f$\pm$%.2f'%(np.mean(sta[:,0]),np.std(sta[:,0])),'%.2f$\pm$%.2f'%(np.mean(sta[:,1]),np.std(sta[:,1])),'%.2f$\pm$%.2f'%(np.mean(sta[:,2]),np.std(sta[:,2])),'%.2f$\pm$%.2f'%(np.mean(sta[:,3]),np.std(sta[:,3])),'%.2f$\pm$%.2f'%(np.mean(sta[:,4]),np.std(sta[:,4])),'%.2f$\pm$%.2f'%(np.mean(sta[:,5]),np.std(sta[:,5]))]]
# table = Texttable()
# table.set_cols_align(["c"] * len(rows[0]))
# table.set_deco(Texttable.HEADER | Texttable.VLINES)
# table.set_cols_dtype(["t", "t","t", "t","t", "t","t"])#This is becouse is was changig the format to 3 decimal palces for some reason
# table.add_rows(rows)
# 
# print(latextable.draw_latex(table, caption="Recovering 0 Myr evolved Arches."
#                             ,use_booktabs=True, caption_above=True,label ='Table_recovered'))
# # print(tabulate(rows, headers='firstrow', tablefmt="latex_booktabs"))
# 
# # %%
# 
# rows = [['Cluster','$\mu_{ra}$','$\mu_{dec}$','$\sigma_{\mu dec}$','$\sigma_{\mu dec}$', 'Radio','stars'],
#         ['Model',str('%.2f'%(model[0])),'%.2f'%(model[1]),'%.2f'%(model[2]),'%.2f'%(model[3]),'%.2f'%(model[4]),'%.2f'%(model[5])],
#         ['Recovered','%.2f$\pm$%.2f'%(np.mean(sta[:,0]),np.std(sta[:,0])),'%.2f$\pm$%.2f'%(np.mean(sta[:,1]),np.std(sta[:,1])),'%.2f$\pm$%.2f'%(np.mean(sta[:,2]),np.std(sta[:,2])),'%.2f$\pm$%.2f'%(np.mean(sta[:,3]),np.std(sta[:,3])),'%.2f$\pm$%.2f'%(np.mean(sta[:,4]),np.std(sta[:,4])),'%.2f$\pm$%.2f'%(np.mean(sta[:,5]),np.std(sta[:,5]))]]
# table = Texttable()
# table.set_cols_align(["c"] * len(rows[0]))
# table.set_deco(Texttable.HEADER | Texttable.VLINES)
# table.set_cols_dtype(["t", "t","t", "t","t", "t","t"])#This is becouse is was changig the format to 3 decimal palces for some reason
# table.add_rows(rows)
# 
# print(latextable.draw_latex(table, caption="\textcolor{red}{MAKE the fucking dissolve cluster simulation}"
#                             ,use_booktabs=True, caption_above=True,label ='Table_disolved'))
# # print(tabulate(rows, headers='firstrow', tablefmt="latex_booktabs"))
# # %%
# # Values of runnig dbscab on Hoseck Arches and Quintuplet (not thorwn in Lib data)
# pruebas_hos = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
# # pm_ra pm_dec sig_pmra sig_dec radio eff_radio #stars
# Ar_dbs= np.loadtxt(pruebas_hos +  'Arches_dbs_statistic.txt',unpack = False)
# Qu_dbs = np.loadtxt(pruebas_hos +  'Quintuplet_dbs_statistic.txt')
# 
# rows = [['Cluster','$\mu_{ra}$','$\mu_{dec}$','$\sigma_{\mu dec}$','$\sigma_{\mu dec}$', 'Radio','Radio$_{eff}$','stars'],
#         ['Arches',
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,0]),np.std(Ar_dbs[:,0])),
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,1]),np.std(Ar_dbs[:,1])),
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,2]),np.std(Ar_dbs[:,2])),
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,3]),np.std(Ar_dbs[:,3])),
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,4]),np.std(Ar_dbs[:,4])),
#          '%.2f$\pm$%.2f'%(np.mean(Ar_dbs[:,5]),np.std(Ar_dbs[:,5])),
#          '%.0f$\pm$%.0f'%(np.mean(Ar_dbs[:,6]),np.std(Ar_dbs[:,6]))],
#         ['Quintuplet',
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,0]),np.std(Qu_dbs[:,0])),
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,1]),np.std(Qu_dbs[:,1])),
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,2]),np.std(Qu_dbs[:,2])),
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,3]),np.std(Qu_dbs[:,3])),
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,4]),np.std(Qu_dbs[:,4])),
#          '%.2f$\pm$%.2f'%(np.mean(Qu_dbs[:,5]),np.std(Qu_dbs[:,5])),
#          '%.0f$\pm$%.0f'%(np.mean(Qu_dbs[:,6]),np.std(Qu_dbs[:,6]))]]
# 
# table = Texttable()
# table.set_cols_align(["c"] * len(rows[0]))
# table.set_deco(Texttable.HEADER | Texttable.VLINES)
# # table.set_cols_dtype(["t", "t","t", "t","t", "t","t"])#This is becouse is was changig the format to 3 decimal palces for some reason
# table.add_rows(rows)
# 
# print(latextable.draw_latex(table, caption="Dbscan on Hosek data"
#                             ,use_booktabs=True, caption_above=True,label ='Table_no_disolved'))
# 
# 
# =============================================================================





