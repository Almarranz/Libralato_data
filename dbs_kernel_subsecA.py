#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:28:52 2022

@author: amartinez
"""

# =============================================================================
# Here we are going to divide section A in smalles LxL areas, thar overlap. Then
# we´ll run dbs with the kernel method over the first of thes boxes, store the cluster 
# if we like in a particular folder called 'cluster1' and continue with the nex box.
# If we found the same (or partially the same) cluster in an overlapping box, we will store it
# in the same folder 'cluster 1', an so on
# =============================================================================
# %%imports
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pylab as p
from random import seed
from random import random
import glob
from sklearn.preprocessing import StandardScaler
import os
import math
from scipy.stats import gaussian_kde
import shutil
from datetime import datetime
import astropy.coordinates as ap_coor

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

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
# catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))

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
color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)
# %%
sec_clus = pruebas +'Sec_A_clus/'
ifE_sec = os.path.exists(sec_clus)
if not ifE_sec:
    os.makedirs(pruebas + 'Sec_%s_clus/'%(section))

carp_clus = sec_clus +'/cluster_num*'


clus_to_erase = glob.glob(pruebas + 'Sec_%s_clus/'%(section)+'cluster_num*')
for f_e in range(len(clus_to_erase)):
    # print(clus_to_erase)
    shutil.rmtree(clus_to_erase[f_e], ignore_errors=True)

# %
# clus_test = np.loadtxt(pruebas + 'dbs_cluster1_of_group89.txt')
m1 = -0.80
m = 1
step = 3300


#This for removin previous subsections
# for f_remove in glob.glob(pruebas + 'subsec_%s/subsec*'%(section)):
#     os.remove(f_remove)

missing =0
# fig, ax = plt.subplots(1,1, figsize=(10,10))
# ax.scatter(catal[:,7],catal[:,8])
fila =-1
lim_pos_up, lim_pos_down = 26700, 18500 #intersection of the positives slopes lines with y axis,
lim_neg_up, lim_neg_down =32700,26000 #intersection of the negayives slopes lines with y axis,

# =============================================================================
# yg_up =  lim_pos_up + m*catal[:,7]
# yg_down =  lim_pos_down + m*catal[:,7]
# 
# =============================================================================
# distancia entre yg_up e yg_down
dist_pos = abs((-1*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))

# =============================================================================
# yr_up = lim_neg_up + m1*catal[:,7]
# yr_down = lim_neg_down + m1*catal[:,7]
# =============================================================================
# distancia entre yg_up e yg_down
dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
ang = math.degrees(np.arctan(m1))

clus_num = 0
x_box = 1
step = dist_pos /x_box
step_neg =dist_neg/x_box

clustered_by = 'all_color'
# samples_dist = 5
x_box_lst = [3,2,1]
samples_lst =[5,7,10]
for x_box in x_box_lst:
    for samples_dist in samples_lst:
        for ic in range(x_box*2-1):
            
            
            ic *= 0.5
            yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
            # yg_2 = (lim_pos_up - (ic+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
            yg_2 = (lim_pos_up - (ic+1)*step/np.cos(45*u.deg)) +  m*catal[:,7]
        
            # ax.plot(catal[:,7],yg_1, color ='g')
            # ax.plot(catal[:,7],yg_2, color ='g')
            for jr in range(x_box*2-1):
                fig, ax = plt.subplots(1,1, figsize=(10,10))
                ax.scatter(catal[:,7],catal[:,8],alpha =0.01)
                jr *=0.5
                yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                        & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
                area = step*step_neg*0.05**2/3600
                
                
                ax.scatter(catal[:,7][good],catal[:,8][good],color =strin[np.random.choice(indices)],alpha = 0.1)
                
                # ax.plot(catal[:,7],yr_1, color ='r')
                # ax.plot(catal[:,7],yr_2, color ='r')
                props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                # place a text box in upper left in axes coords
                txt ='central box ~ %.1f arcmin$^{2}$'%(area)
                ax.text(0.65, 0.95, txt, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
                ax.set_xlabel('x (50 mas/pix)')
                ax.set_ylabel('y (50 mas/pix)')
        # =============================================================================
        #         Here is where the party begins
        # =============================================================================
                datos = catal[good]
                
                # % coordinates
                ra_=datos[:,5]
                dec_=datos[:,6]
                # Process needed for the trasnformation to galactic coordinates
                coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
                gal_c=coordenadas.galactic
        
                t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))
                
                mul,mub = datos[:,-6],datos[:,-5]
                x,y = datos[:,7], datos[:,8]
                colorines = datos[:,3]-datos[:,4]
                
                
                mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
                x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
                color_kernel = gaussian_kde(colorines)
                if clustered_by == 'all_color':
                    X=np.array([mul,mub,datos[:,7],datos[:,8],colorines]).T
                    X_stad = StandardScaler().fit_transform(X)
                    tree = KDTree(X_stad, leaf_size=2) 
                    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
                elif clustered_by == 'all':
                    X=np.array([mul,mub,datos[:,7],datos[:,8]]).T
                    X_stad = StandardScaler().fit_transform(X)
                    tree = KDTree(X_stad, leaf_size=2) 
                    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
                # For the simulated data we loop a number of times and get the average of the minimun distance
                lst_d_KNN_sim = []
                for d in range(20):
                    mub_sim,  mul_sim = mub_kernel.resample(len(datos)), mul_kernel.resample(len(datos))
                    x_sim, y_sim = x_kernel.resample(len(datos)), y_kernel.resample(len(datos))
                    color_sim = color_kernel.resample(len(datos))
                    if clustered_by == 'all_color':
                        X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                        X_stad_sim = StandardScaler().fit_transform(X_sim)
                        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                        
                        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                        
                        lst_d_KNN_sim.append(min(d_KNN_sim))
                    elif clustered_by =='all':
                        X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0]]).T
                        X_stad_sim = StandardScaler().fit_transform(X_sim)
                        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                        
                        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                        
                        lst_d_KNN_sim.append(min(d_KNN_sim))
                
                d_KNN_sim_av = np.mean(lst_d_KNN_sim)
                
        
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                # ax[0].set_title('Sub_sec_%s_%s'%(col[colum],row[ro]))
                # ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
                # ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')
                
                # # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
                # ax[0].set_xlabel('Point') 
                # ax[0].set_ylabel('%s-NN distance'%(samples)) 
                
                ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
                ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
                ax.set_xlabel('%s-NN distance'%(samples_dist)) 
                
                eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
                texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                                    'min sim d_KNN =%s'%(round(d_KNN_sim_av,3)),'average = %s'%(eps_av)))
                
        
                props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                # place a text box in upper left in axes coords
                ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
                    verticalalignment='top', bbox=props)
                
                ax.set_ylabel('N') 
                # ax.set_xlim(0,1)
               
               
                clus_method = 'dbs'
        
                clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
                l=clustering.labels_
                
                n_clusters = len(set(l)) - (1 if -1 in l else 0)
                # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
                n_noise=list(l).count(-1)
                # %
                u_labels = set(l)
                colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
                # %
                
                # %
                for k in range(len(colors)): #give noise color black with opacity 0.1
                    if list(u_labels)[k] == -1:
                        colors[k]=[0,0,0,0.1]
                # %      
                colores_index=[]
                
                for c in u_labels:
                    cl_color=np.where(l==c)
                    colores_index.append(cl_color)
                
                
                # %
                for i in range(len(set(l))-1):
                    fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # ax[2].invert_yaxis()
                   
                    ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
                    # t_gal['l'] = t_gal['l'].wrap_at('180d')
                    ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
                    ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
                    # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
            
                    ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color='fuchsia',s=50,zorder=3)
                    ax[0].set_xlim(-10,10)
                    ax[0].set_ylim(-10,10)
                    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
                    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
                    ax[0].invert_xaxis()
                    ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
                    
                    mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
                    mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
                    
                    mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
                    mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
                
                
                    vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
                    vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
                    
                    propiedades = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
                    propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
                    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades)
                    ax[0].text(0.05, 0.85, vel_txt_all, transform=ax[0].transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades_all)
                    
                    radio = 500*u.arcsec
                    
                    
                    #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
                    c2 = SkyCoord(ra = datos[:,0][colores_index[i]]*u.deg,dec = datos[:,1][colores_index[i]]*u.deg)
                    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
                    rad = max(sep)/2
                    
                    radio_MS = max(sep)
                    
                    prop = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
                    ax[1].text(0.65, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=14,
                                            verticalalignment='top', bbox=prop)
                    
                    ax[1].scatter(catal[:,7], catal[:,8], color='k',s=50,zorder=1,alpha=0.01)#plots in galactic
                    ax[1].scatter(X[:,2], X[:,3], color=colors[-1],s=50,zorder=1,alpha=0.02)#plots in galactic
                    
                    ax[1].scatter(X[:,2][colores_index[i]], X[:,3][colores_index[i]], color='blueviolet',s=50,zorder=3)#plots in galactic
                    ax[1].quiver(X[:,2][colores_index[i]], X[:,3][colores_index[i]], X[:,0][colores_index[i]]*-1, X[:,1][colores_index[i]], alpha=0.5, color='blueviolet')#colors[i]
                    ax[1].set_xlabel('x') 
                    ax[1].set_ylabel('y') 
                    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax[1].set_title('col_row %.0f, %.0f'%(ic/0.5,jr/0.5))
                    # This is for plotting the cluster and the isochrone with extiontion
                    # here look for the values of extintiction for the cluster stars in the extintion catalog
                    # if does not have a value it will not plot the star in the CMD
                    H_Ks_yes = []
                    Ks_yes = []
                    AKs_clus_all =[]
                    
                    clus_coord =  SkyCoord(ra=datos[:,5][colores_index[i][0]]*u.degree, dec=datos[:,6][colores_index[i][0]]*u.degree)
                    idx = clus_coord.match_to_catalog_sky(gns_coord)
                    gns_match = AKs_center[idx[0]]
                    for member in range(len(gns_match)):
                        if gns_match[member,16] != '-' and gns_match[member,18] != '-':
                            AKs_clus_all.append(float(gns_match[member,18]))
                            H_Ks_yes.append(datos[:,3][colores_index[i][0][member]]-datos[:,4][colores_index[i][0][member]])
                            Ks_yes.append(datos[:,4][colores_index[i][0][member]])
                    print(ic, jr, len(mul),n_clusters)   
                    ax[2].scatter(H_Ks_yes,Ks_yes, color='blueviolet',s=50,zorder=3, alpha=1)
                    ax[2].invert_yaxis()  
                    
                    AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                    # absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                    # AKs = min(AKs_list, key=absolute_difference_function)
                    
                    # iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
                    
                    # dist = 8000 # distance in parsec
                    # metallicity = 0.17 # Metallicity in [M/H]
                    # # logAge_600 = np.log10(0.61*10**9.)
                    # logAge = np.log10(0.010*10**9.)
                    # logAge_30 = np.log10(0.030*10**9.)
                    # logAge_60 = np.log10(0.060*10**9.)
                    # logAge_90 = np.log10(0.090*10**9.)
                    # evo_model = evolution.MISTv1() 
                    # atm_func = atmospheres.get_merged_atmosphere
                    # red_law = reddening.RedLawNoguerasLara18()
                    # filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
                    
                    # iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                    #                                 evo_model=evo_model, atm_func=atm_func,
                    #                                 red_law=red_law, filters=filt_list,
                    #                                     iso_dir=iso_dir)
                    
                    # iso_30 = synthetic.IsochronePhot(logAge_30, AKs, dist, metallicity=metallicity,
                    #                                 evo_model=evo_model, atm_func=atm_func,
                    #                                 red_law=red_law, filters=filt_list,
                    #                                     iso_dir=iso_dir)
                    # iso_60 = synthetic.IsochronePhot(logAge_60, AKs, dist, metallicity=metallicity,
                    #                                 evo_model=evo_model, atm_func=atm_func,
                    #                                 red_law=red_law, filters=filt_list,
                    #                                     iso_dir=iso_dir)
                    
                    # iso_90 = synthetic.IsochronePhot(logAge_90, AKs, dist, metallicity=metallicity,
                    #                                 evo_model=evo_model, atm_func=atm_func,
                    #                                 red_law=red_law, filters=filt_list,
                    #                                     iso_dir=iso_dir)
                    # # #%
                    # #%
                    
                    
                    # imf_multi = multiplicity.MultiplicityUnresolved()
                    
                    # # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01
                    
                    # # NOTE: when defining the power law slope for each segment of the IMF, we define
                    # # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
                    # # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 
                    
                    # massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
                    # powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
                    # # my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
                    # my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)
                    
                    
                    # #%
                    
                    
                    # mass = 0.5*10**4.
                    # mass = 1 * mass
                    # dAks = round(std_AKs*1,3)
                    # cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
                    # cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
                    # clus = cluster.star_systems
                    # clus_ndiff = cluster_ndiff.star_systems
                    ax[2].set_title('Cluster %s, eps = %s'%(i,round(eps_av,3)))
                    ax[2].scatter(datos[:,3]-datos[:,4],datos[:,4],alpha=0.1)
                    ax[2].set_xlim(1.3,2.5)
                    # ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=0.1)
                    # ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.1,s=1)
                    
        
                    # txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
                    #                      'age = %.0f Myr'%(10**logAge/10**6)))
                    txt_AKs = '\n'.join(('H-Ks =%.3f'%(np.mean(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
                                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
                                         'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))
                                         ,'AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    # # place a text box in upper left in axes coords
                    ax[2].text(0.65, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    # ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
                    #     verticalalignment='top', bbox=props)
                    # ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
                    #                   iso.points['m_hawki_Ks'], 'b-',  label='10 Myr')
                    
                    # ax[2].plot(iso_30.points['m_hawki_H'] - iso_30.points['m_hawki_Ks'], 
                    #                   iso_30.points['m_hawki_Ks'], 'orange',  label='30 Myr')
                    # ax[2].plot(iso_60.points['m_hawki_H'].value - iso_60.points['m_hawki_Ks'].value, 
                    #                   iso_60.points['m_hawki_Ks'].value, color ='green' ,label='60 Myr')
                    # ax[2].plot(iso_90.points['m_hawki_H'] - iso_90.points['m_hawki_Ks'], 
                    #                   iso_90.points['m_hawki_Ks'], 'red' ,label='90 Myr')
                    # ax[2].set_xlabel('H$-$Ks')
                    # ax[2].set_ylabel('Ks')
                    # ax[2].legend(loc =3, fontsize = 12)
                    plt.show()
                    clus_array = np.array([datos[:,5][colores_index[i]],datos[:,6][colores_index[i]],t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value,
                                                                                          X[:,0][colores_index[i]], 
                                                                                          X[:,1][colores_index[i]],
                                                                                          datos[:,2][colores_index[i]],datos[:,3][colores_index[i]],datos[:,4][colores_index[i]],
                                                                                         datos[:,7][colores_index[i]],datos[:,8][colores_index[i]]]).T
                    
        # =============================================================================
        #             Here it compare the cluster you want to save wiith the rest of the 
        #             saved cluster if repited, it saves in the same cluster
        #             
        # =============================================================================
                    
                    frase = 'Do you want to save this cluster?'
                    print('\n'.join((len(frase)*'π',frase+'\n("yes" or "no")',len(frase)*'π')))
                    save_clus = input('Awnser:')
                    print('You said: %s'%(save_clus))
                    if save_clus =='yes' or save_clus =='y':
                        intersection_lst =[]
                        clus_num +=1
                        check_folder = glob.glob(pruebas + 'Sec_%s_clus/'%(section)+'cluster_num*')
                        if len(check_folder) == 0:
                            os.makedirs(pruebas + 'Sec_%s_clus/'%(section) +'cluster_num%s_%s_knn%s_area%.2f/'%(clus_num,i,samples_dist,area))
                            np.savetxt(pruebas + 'Sec_%s_clus/'%(section) +'cluster_num%s_%s_knn%s_area%.2f/'%(clus_num,i,samples_dist,area)+
                                       'cluster%s_%.0f_%.0f_knn_%s_area_%.2f.txt'%(clus_num,ic/0.5,jr/0.5,samples_dist,area),clus_array,
                                       fmt='%.7f '*6 + ' %.4f'*3 +' %.5f'*2,
                                       header ='ra, dec, l, b, pml, pmb,J, H, Ks,x, y')
                        else:
                            for f_check in check_folder:
                                
                                clus_lst = os.listdir(f_check)
                                for n_txt in clus_lst:
                                    ra_dec = np.loadtxt(f_check+'/'+n_txt,usecols=(0,1))
                                    ra_dec_clus = clus_array[:,0:2]
                                    aset = set([tuple(x) for x in ra_dec_clus])
                                    bset = set([tuple(x) for x in ra_dec])
                                    intersection = np.array([x for x in aset & bset])
                                    intersection_lst.append(len(intersection))
                                    print('This is intersection',intersection_lst)
                                    if len(intersection)> 0 :
                                        print('Same (or similar) cluster  is in %s'%(f_check))
                                        np.savetxt(f_check+'/'+
                                                   'cluster%s_%.0f_%.0f_knn_%s_area_%.2f.txt'%(clus_num,ic/0.5,jr/0.5,samples_dist,area),clus_array,
                                                   fmt='%.7f '*6 + ' %.4f'*3 +' %.5f'*2,
                                                   header ='ra, dec, l, b, pml, pmb,J, H, Ks,x, y')
                            if np.all(np.array(intersection_lst)==0):
                                # clus_num +=1
                                print('NEW CLUSTER')
                                os.makedirs(pruebas + 'Sec_%s_clus/'%(section) +'cluster_num%s_%s_knn%s_area%.2f/'%(clus_num,i,samples_dist,area))
                                np.savetxt(pruebas + 'Sec_%s_clus/'%(section) +'cluster_num%s_%s_knn%s_area%.2f/'%(clus_num,i,samples_dist,area)+
                                           'cluster%s_%.0f_%.0f_knn_%s_area_%.2f.txt'%(clus_num,ic/0.5,jr/0.5,samples_dist,area),clus_array,
                                           fmt='%.7f '*6 + ' %.4f'*3 +' %.5f'*2,
                                           header ='ra, dec, l, b, pml, pmb,J, H, Ks,x, y')
                                
                                    
                                # read_txt = glob.glob(check_folder[f_check]+'/cluster_*')
                                # for clust_text in range(len(read_txt)):
                                #     print(read_txt[clust_text])
                    elif save_clus =='stop':
                        frase = 'Do you want to copy the folder with the clusters into the morralla directory?\n("yes" or "no")'
                        print('\n'.join((len(frase)*'',frase,len(frase)*'')))
                        save_folder = input('Awnser:')   
                        if save_folder == 'yes' or save_folder == 'y':       
                            source_dir = pruebas + 'Sec_%s_clus/'%(section)
                            destination_dir = '/Users/amartinez/Desktop/morralla/Sec_%s_at_%s'%(section,datetime.now())
                            shutil.copytree(source_dir, destination_dir)
                            sys.exit('You stoped it')
                        else:
                            sys.exit('You stoped it')
                        sys.exit('Chao')
                    
                    else:
                        print('No saved')
                
frase = 'Do you want to copy the folder with the clusters into the morralla directory?\n("yes" or "no")'
print('\n'.join((len(frase)*'',frase,len(frase)*'')))
save_folder = input('Awnser:')   
if save_folder == 'yes' or save_folder == 'y':       
    source_dir = pruebas + 'Sec_%s_clus/'%(section)
    destination_dir = '/Users/amartinez/Desktop/morralla/Sec_%s_at_%s'%(section,datetime.now())
    shutil.copytree(source_dir, destination_dir)
    sys.exit('You stoped it')
else:
    sys.exit('You stoped it')
sys.exit('Chao')
      
      
      
      
      

