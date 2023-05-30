#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:18:16 2023

@author: amartinez
"""
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

# Lib catalog indices
# "'RA_gns' 0	DE_gns' 1	Jmag' 2	Hmag' 3	Ksmag' 4	ra' 5	dec' 6	x_c' 7	y_c' 8	
# mua' 9	dmua' 10	mud' 11	dmud' 12	time' 13	n1' 14	n2' 15	ID' 16	mul' 17	mub' 18	
# dmul' 19	dmub' 20	m139' 21	Separation'" 22		

def dbscan_GC(pmra, pmdec, x, y,Ra,Dec, color_A, color_B, clustered_by, samples_dist,Ms_match,gen_sim, sim_lim):
    mix_color = 'yes'
    
    coordenadas = SkyCoord(ra = Ra,dec=Dec, unit='degree')
    colorines = color_A-color_B
    pmra_kernel, pmdec_kernel = gaussian_kde(pmra), gaussian_kde(pmdec)
    x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
    color_kernel = gaussian_kde(colorines)
    if clustered_by == 'all_color':
        X = np.array([pmra,pmdec,x,y,colorines]).T
        X = np.array([pmra,pmdec,x,y,colorines]).T
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    elif clustered_by == 'all':
        X = np.array([pmra,pmdec,x, y]).T
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
    if gen_sim == 'kernnel':
        for d in range(20):
            mudec_sim,  mura_sim = pmdec_kernel.resample(len(pmdec)), pmra_kernel.resample(len(pmra))
            x_sim, y_sim = x_kernel.resample(len(x)), y_kernel.resample(len(y))
            color_sim = color_kernel.resample(len(pmdec))
            if clustered_by == 'all_color':
                X_sim=np.array([mura_sim[0],mudec_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                
                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by =='all':
                X_sim=np.array([mura_sim[0],mudec_sim[0],x_sim[0],y_sim[0]]).T
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
    if gen_sim == 'shuffle':
        for d in range(5):
            randomize = np.arange(len(pmdec))
            np.random.shuffle(randomize)
            mudec_sim,  mura_sim= pmdec[randomize], pmra[randomize]
            x_sim, y_sim = x, y
            
            random_col = np.arange(len(pmdec))
            np.random.shuffle(random_col)
            if mix_color == 'yes':
                H_sim, K_sim = color_A[random_col], color_B[random_col]
            elif mix_color == 'no':
                H_sim, K_sim = color_A, color_B
            color_sim = H_sim-K_sim
            if clustered_by == 'all_color':
                X_sim=np.array([mura_sim,mudec_sim,x_sim,y_sim,color_sim]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                
                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by =='all':
                X_sim=np.array([mura_sim,mudec_sim,x_sim,y_sim]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                
                lst_d_KNN_sim.append(min(d_KNN_sim))
            elif clustered_by =='vel_col':
                X_sim=np.array([mura_sim,mudec_sim,color_sim]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                
                lst_d_KNN_sim.append(min(d_KNN_sim))
                
            elif clustered_by =='vel':
                X_sim=np.array([mura_sim,mudec_sim]).T
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                
                lst_d_KNN_sim.append(min(d_KNN_sim))
    d_KNN_sim_av = np.mean(lst_d_KNN_sim)
    if sim_lim == 'mean':
        eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
        valor = d_KNN_sim_av
    elif sim_lim == 'minimun':
        eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2,3)
        valor = min(lst_d_KNN_sim)
    elif sim_lim == 'maximun':
        eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2,3)
        valor = min(lst_d_KNN_sim)
    
    
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
        
    for i in range(len(set(l))-1):
        ms_in_clus = np.where((Ra[colores_index[i]] == Ms_match[0]) & (Dec[colores_index[i]] == Ms_match[1]))
        if len(ms_in_clus[0]) > 0:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
            ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
            ax.set_xlabel('%s-NN distance'%(samples_dist)) 
            
            
            texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                                'limit set for sim d_KNN =%s'%(round(valor,3)),
                                'average = %s'%(eps_av),'%s'%(sim_lim),'%s'%(gen_sim)))

            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)
            
            ax.set_ylabel('N') 
            fig, ax = plt.subplots(1,3,figsize=(30,10))
            color_de_cluster = 'lime'
            # fig, ax = plt.subplots(1,3,figsize=(30,10))
            # ax[2].invert_yaxis()
           
            ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
            # t_gal['l'] = t_gal['l'].wrap_at('180d')
            ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
            
            ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
            # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
            ax[0].scatter(Ms_match[9], Ms_match[11], s=200, c ='r')
            ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=color_de_cluster ,s=50,zorder=3)
            ax[0].set_xlim(-10,10)
            ax[0].set_ylim(-10,10)
            ax[0].set_xlabel(r'$\mathrm{\mu_{ra} (mas\ yr^{-1})}$',fontsize =30) 
            ax[0].set_ylabel(r'$\mathrm{\mu_{dec} (mas\ yr^{-1})}$',fontsize =30) 
            ax[0].invert_xaxis()
            ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
            
            mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
            mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
            
            mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
            mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
        
        
            vel_txt = '\n'.join(('mura = %s, mudec = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                  '$\sigma_{mura}$ = %s, $\sigma_{mudec}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
            vel_txt_all = '\n'.join(('mura = %s, mudec = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                  '$\sigma_{mura}$ = %s, $\sigma_{mudec}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
            
            propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
            propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
            ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
                verticalalignment='top', bbox=propiedades)
            ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
                verticalalignment='top', bbox=propiedades_all)
        
            #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
            c2 = SkyCoord(ra = Ra[colores_index[i]]*u.deg,dec = Dec[colores_index[i]]*u.deg)
            sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
            rad = max(sep)/2
            
            radio_MS = max(sep)
            
            m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)])
            
            idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*2)
            
            ax[0].scatter(Ra[group_md],Dec[group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)
    
            prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
            ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
                                    verticalalignment='top', bbox=prop)
            
            ax[1].scatter(Ms_match[0], Ms_match[1], s=200, c='r')
    
            # ax[1].scatter(catal[:,7], catal[:,8], color='k',s=50,zorder=1,alpha=0.01)#plots in galactic
            ax[1].scatter(Ra, Dec, color=colors[-1],s=50,zorder=1,alpha=0.2)#plots in galactic
            
            ax[1].scatter(Ra[colores_index[i]], Dec[colores_index[i]], color=color_de_cluster ,s=50,zorder=3)#plots in galactic
            ax[1].quiver(Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]], alpha=1, color='black' )#colors[i]
            # ax[1].scatter(datos[:,7][group_md],datos[:,8][group_md],s=50,color='r',alpha =0.1,marker ='x')
            ax[1].set_xlabel('Ra',fontsize =30) 
            ax[1].set_ylabel('Dec',fontsize =30) 
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            
            
            # ms_in_clus = np.where((Ra[colores_index[i]] == Ms_match[0]) & (Dec[colores_index[i]] == Ms_match[1]))
            # ms_in_clus = np.where((Ra[colores_index[i]] == Ra[colores_index[i]][0]) & (Dec[colores_index[i]] == Dec[colores_index[i]][0]))
    
            if len(ms_in_clus[0]) > 0:
                # ax[1].set_facecolor('lavender')
                print(12*'*'+'\nSOMETHIG!!!!\n'+12*'*')
                ax[1].set_facecolor('mistyrose')
            ax[2].scatter(Ms_match[3] - Ms_match[4], Ms_match[4], s=200,c ='r')    
            ax[2].scatter(color_A-color_B,color_B, alpha=0.1,color ='k')
            # ax[2].scatter(color_A[group_md]-color_B[group_md], color_B[group_md],alpha=0.7,c='r',marker = 'x')
            txt_around = '\n'.join(('H-Ks =%.3f'%(np.median(color_A[group_md]-color_B[group_md])),
                                 '$\sigma_{H-Ks}$ = %.3f'%(np.std(color_A[group_md]-color_B[group_md])),
                                 'diff_color = %.3f'%(max(color_A[group_md]-color_B[group_md])-min(color_A[group_md]-color_B[group_md]))))
            props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
            ax[2].text(0.50, 0.25,txt_around, transform=ax[2].transAxes, fontsize=30,
                verticalalignment='top', bbox=props_arou)
    
            ax[2].scatter(color_A[colores_index[i]]-color_B[colores_index[i]], color_B[colores_index[i]],alpha=1,c='lime')
    
    
            # txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
            #                      'age = %.0f Myr'%(10**logAge/10**6)))
            txt_color = '\n'.join(('H-Ks =%.3f'%(np.median(color_A[colores_index[i]]-color_B[colores_index[i]])),
                                 '$\sigma_{H-Ks}$ = %.3f'%(np.std(color_A[colores_index[i]]-color_B[colores_index[i]])),
                                 'diff_color = %.3f'%(max(color_A[colores_index[i]]-color_B[colores_index[i]])-min(color_A[colores_index[i]]-color_B[colores_index[i]]))))
            props = dict(boxstyle='round', facecolor=color_de_cluster, alpha=0.3)
            # # place a text box in upper left in axes coords
            ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
                verticalalignment='top', bbox=props)
            # ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
            #     verticalalignment='top', bbox=props)
            ax[2].set_xlabel('H - Ks',fontsize =30)
            ax[2].set_ylabel('Ks',fontsize =30)
            ax[2].set_xlim(1.3,2.5)
            ax[2].invert_yaxis()   
            print('+++++++++++++++')
            print(len(ms_in_clus[0]))
            print('+++++++++++++++')
            # return len(ms_in_clus[0]) 
            
            print(type(ms_in_clus[0]), type(Ra[colores_index[i]]), 
                  type(Dec[colores_index[i]]), type(X[:,0][colores_index[i]]),
                  type(X[:,1][colores_index[i]]))
            if len(ms_in_clus[0]) >0:
                return len(ms_in_clus[0]), Ra[colores_index[i]], Dec[colores_index[i]], X[:,0][colores_index[i]],X[:,1][colores_index[i]]
            else:
                return [0,0,0,0,0]
        
# %%
# lis, oli = 300000, 70000
# val = 0.3
# coef = 0.2
# her = 5

# pagar = lis*(1*coef*val)/her + oli*(1*coef*val)/her
    
# print(pagar)
    
    