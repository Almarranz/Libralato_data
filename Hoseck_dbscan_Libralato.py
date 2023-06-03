#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:27:52 2022

@author: amartinez
"""

# Here we are gooing to look for the Arches cluster using dbscan
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

# Here we are gooing to look for the Arches cluster using dbscan
# %% Here we are going to trimm the data
# Only data with valid color and uncertainties in pm smaller than 0.4 and excluding foreground stars by color-cut
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
morralla ='/Users/amartinez/Desktop/morralla/'
cls_to_throw = '/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/'

choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

ref_frame = 'ecuatorial'#TODO
# ref_frame = 'galactic'#TODO

cluster_gone = 'no'
center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet
# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>

if choosen_cluster == 'Arches':
    arches = ascii.read(catal + 'Arches_from_Article.txt') 
if choosen_cluster == 'Quintuplet':
    arches = ascii.read(catal + 'Quintuplet_from_Article.txt') 
    
    

m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]
arches=arches[valid_colors]

center = np.where((m127.value - m153.value > 1.7))
arches = arches[center]




pme_lim = 1.4
valid_epm = np.where((arches['e_pmRA'] < pme_lim)&(arches['e_pmDE'] < pme_lim))
arches=arches[valid_epm]

# DEC = center_arc.dec.value + arches['dDE']*u.arcsec.to(u.degree)
# RA =center_arc.ra.value +(arches['dRA']*u.arcsec.to(u.degree)/np.cos(DEC))
RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
RA = RA_DEC.ra
DEC = RA_DEC.dec

if cluster_gone == 'yes':
    m_arc = SkyCoord(ra = [266.46081294], dec = [-28.82303557],unit = 'degree' )
    ar_coor = SkyCoord(ra =  RA_DEC.ra, dec =  RA_DEC.dec)
    idxc, around_arc, d2d,d3d =  ap_coor.search_around_sky(m_arc, ar_coor, 30*u.arcsec)
    arches.remove_rows([around_arc])
    RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
    RA = RA_DEC.ra
    DEC = RA_DEC.dec
    

# sys.exit('98')
# Galactic coordinates
arches_gal = SkyCoord(ra=RA, dec=DEC,
                    pm_ra_cosdec =arches['pmRA'], pm_dec = arches['pmDE'],
                    unit = 'degree',frame = 'icrs', obstime='J2016.0').galactic

arches.add_column(arches_gal.pm_l_cosb, name = 'pml',index =-1)
arches.add_column(arches_gal.pm_b,name = 'pmb',index = -1)

if ref_frame == 'ecuatorial':
    pmra, pmdec = arches['pmRA'], arches['pmDE']
if ref_frame == 'galactic':
    pmra, pmdec = arches['pml'], arches['pmb']

raoff, decoff = arches['dRA'], arches['dDE']
colorines = arches['F127M']-arches['F153M']
pmra_kernel, pmdec_kernel = gaussian_kde(arches['pmRA']), gaussian_kde(arches['pmDE'])
raoff_kernel, decoff_kernel = gaussian_kde(raoff), gaussian_kde(decoff)
color_kernel = gaussian_kde(colorines)


clustered_by = 'all_color'#TODO
# clustered_by = 'all'#TODO1
# clustered_by = 'vel_col'#TODO


samples_dist = 40
RA_ = np.array(RA.value)
DEC_ = np.array(DEC.value)

if clustered_by == 'all_color':
    # X = np.array([pmra,pmdec,raoff,decoff,colorines]).T
    X = np.array([pmra,pmdec,RA,DEC,colorines]).T
    X_stad = StandardScaler().fit_transform(X)
    tree = KDTree(X_stad, leaf_size=2) 
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
elif clustered_by == 'all':
    X = np.array([pmra,pmdec,raoff,decoff]).T
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

d_KNN_sim_av = np.mean(lst_d_KNN_sim)
# %%

fig, ax = plt.subplots(1,1,figsize=(10,10))
# ax.set_title('Number of points = %s '%(len(pmdec)))

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k', linewidth = 5,label = 'Arches (min. = %.2f)'%(min(d_KNN)))
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', linewidth = 5,alpha = 0.5,label = 'Simulated (min. = %.2f)'%(d_KNN_sim_av))
ax.set_xlabel('%s-NN distance'%(samples_dist)) 
ax.legend()

eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                    'min sim d_KNN =%s'%(round(d_KNN_sim_av,3)),'average = %s'%(eps_av)))

ax.set_xlim(0,1)
props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# place a text box in upper left in axes coords
# ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
#     verticalalignment='top', bbox=props)

ax.set_ylabel('N') 
plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/Libralato/hist_%a.png'%(choosen_cluster),dpi =300)

# %%
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].scatter(pmra, pmdec)
ax[1].scatter(raoff,decoff)
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
    
# %%
# default colors = '#1f77b4', '#ff7f0e'
elements_in_cluster=[]
# for i in range(len(set(l_c))-1):
for i in range(1):
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    # ax[0].set_xticks(np.arange(-12,11))
    # ax[0].grid()
    ax[0].invert_xaxis()
    ax[2].invert_yaxis()
    c2 = SkyCoord(ra = RA[colores_index[i]],dec =DEC[colores_index[i]], unit ='degree',  equinox = 'J2000', obstime = 'J2015.4')
    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
    rad = max(sep)/2
    ax[0].set_title('%s'%(choosen_cluster))
    # ax[0].scatter(pmra[colores_index[i]], pmdec[colores_index[i]],color=colors[i],zorder=3)
    ax[0].scatter(pmra[colores_index[i]], pmdec[colores_index[i]],color = '#ff7f0e',zorder=3)

    # ax[1].scatter(l[colores_index[i]], b[colores_index[i]],color=colors[i],zorder=3)
    # ax[1].scatter(RA[colores_index[i]],DEC[colores_index[i]],color=colors[i],zorder=3,s=100,alpha =0.3)
    ax[1].scatter(RA[colores_index[i]],DEC[colores_index[i]],color = '#ff7f0e',zorder=3,s=100,alpha =0.3)

    # ax[1].scatter(gns_match[colores_index[i]][:,0],gns_match[colores_index[i]][:,2],color=colors[i],zorder=3,s=100)
    # ax[2].scatter(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]],arches['F153M'][colores_index[i]],color=colors[i],zorder=13)
    ax[2].scatter(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]],arches['F153M'][colores_index[i]],color = '#ff7f0e',zorder=13)

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
    txt_color = '\n'.join(('H-Ks =%.3f'%(np.median(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]])),
                                            '$\sigma_{H-Ks}$ = %.3f'%(np.std(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]]))))
                                            # 'diff_color = %.3f'%(max(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]])-min(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]]))))
    # props = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
    props = dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.2)

    ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
                            verticalalignment='top', bbox=props)
    ax[0].scatter(pmra[colores_index[-1]], pmdec[colores_index[-1]],color=colors[-1],zorder=1)
    
    ax[0].set_xlabel(r'$\mathrm{\mu_{ra*} (mas\ yr^{-1})}$',fontsize =30) 
    ax[0].set_ylabel(r'$\mathrm{\mu_{dec} (mas\ yr^{-1})}$',fontsize =30) 
    # ax[1].scatter(l[colores_index[-1]], b[colores_index[-1]],color=colors[-1],zorder=1)
    ax[1].scatter(RA[colores_index[-1]],DEC[colores_index[-1]],color=colors[-1],zorder=3,s=100,alpha = 0.01)
    # ax[1].scatter(RA,DEC,color=colors[-1],zorder=3,s=100,alpha = 0.01)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
   
    ax[1].set_yticks([-28.85, -28.84, -28.83, -28.81, -28.8 , -28.79])
    ax[1].set_xlabel('ra(deg)',fontsize =30) 
    ax[1].set_ylabel('dec(deg)',fontsize =30)
    ax[1].yaxis.set_label_coords(-0.05, 0.5)
    
    ax[2].scatter(arches['F127M'][colores_index[-1]]-arches['F153M'][colores_index[-1]],arches['F153M'][colores_index[-1]],color=colors[-1],zorder=1)
    ax[2].set_xlabel('f127m-f153m',fontsize =30) 
    ax[2].set_ylabel('f153m',fontsize =30) 
    plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/Libralato/%s_hos.png'%(choosen_cluster),dpi =300)
    # %%
    # Now we ara going to select the cluster stars with prob. of being part of
    # the cluster > 0.7, and then those would be the ones that we will send to 
    # the furute
    arches_prob = arches[colores_index[i]]
    prob_mem = arches_prob['Pclust']
    fig, ax = plt.subplots(1,1)
    ax.hist(prob_mem,bins = 30,label ='mean = %.2f.\n$\sigma$=%.2f'%(np.mean(prob_mem), np.std(prob_mem)))
    ax.legend()
    hi_prob = np.where(prob_mem >= 0.7)
    low_prob = np.where(prob_mem < 0.7)
    # %%
    fig, ax = plt.subplots(1,1)
    # ax.scatter(RA, DEC, color = 'k', alpha = 0.01)
    ax.scatter(RA[colores_index[i]][low_prob], DEC[colores_index[i]][low_prob], color = 'red')
    ax.scatter(RA[colores_index[i]][hi_prob], DEC[colores_index[i]][hi_prob], color = 'blue')
# %%
bins_ =20
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].hist(pmra, bins =bins_, label = '$\overline{\mu_{ra}}$ =%.2f\n$\sigma_{ra}$ =%.2f'%(np.mean(pmra),np.std(pmra)))
ax[0].legend()
ax[1].hist(pmdec, bins =bins_, label = '$\overline{\mu_{dec}}$ =%.2f\n$\sigma_{dec}$ =%.2f'%(np.mean(pmdec),np.std(pmdec)))
ax[1].legend()

ax[0].axvline(np.mean(pmra), color ='red')
ax[1].axvline(np.mean(pmdec), color ='red')
# ax[0].set_xlim(-15,15)
# ax[1].set_xlim(-15,15)
ax[0].set_xlabel('$\mu_{ra}$ (mas/yr)')
ax[1].set_xlabel('$\mu_{dec}$ (mas/yr)')

# %%
# We are going to disolve the cluster and then run dbscan again. The goal is to
# check how dissolve the cluster have to be for be invisible to dbscan.

clus_dbs = np.c_[RA[colores_index[i]].value,DEC[colores_index[i]].value,
                 pmra[colores_index[i]].value, pmdec[colores_index[i]].value,
                 np.array(arches['F127M'][colores_index[i]]),
                 np.array(arches['F153M'][colores_index[i]])]




t = clus_dbs[:,3] / clus_dbs[:,2]

dis_x, dis_y = 0.15, 0.15

# This make movemnet vectors for each star along the dame direction they had, 
# but all with the same module for the velocity
x = np.sqrt(dis_x**2/(1+t**2))*np.sign(clus_dbs[:,2])
y = np.sqrt(dis_y**2 - (dis_y**2/(1+t**2)))*np.sign(clus_dbs[:,3])


cen_RA, cen_DEC = np.median(clus_dbs[:,0])*u.deg, np.median(clus_dbs[:,1])*u.deg
for time in range(1):
    fig, ax = plt.subplots(1,2, figsize = (20,10))
    ax[1].scatter(RA, DEC)
    ax[1].scatter(clus_dbs[:,0], clus_dbs[:,1])
    ax[0].scatter(pmra, pmdec)
    ax[0].scatter(clus_dbs[:,2], clus_dbs[:,3])
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_xlim(min(RA.value),max(RA.value))
    ax[1].set_ylim(min(DEC.value),max(DEC.value))
    
    RA_cl, DEC_cl = clus_dbs[:,0]*u.deg, clus_dbs[:,1]*u.deg



    # ff = 2e5*u.yr  + time*5e5*u.yr
    ff = 0*u.yr  + time*10e5*u.yr
    
    # Balistic dipesion (move each star in a straight line)
    # RA_cl = RA_cl.to(u.mas) - (clus_dbs[:,2]*u.mas/u.yr)*ff
    # DEC_cl = DEC_cl.to(u.mas) + (clus_dbs[:,3]*u.mas/u.yr)*ff
    
    # Sigma dispersion (move each star in the same direction they has but all with the same velocity)
    RA_cl = RA_cl.to(u.mas) - (x*u.mas/u.yr)*ff
    DEC_cl = DEC_cl.to(u.mas) + (y*u.mas/u.yr)*ff
    
    RA_fut, DEC_fut = RA_cl.to(u.deg), DEC_cl.to(u.deg)
    
    cen_RA_fut, cen_DEC_fut = np.median(RA_fut), np.median(DEC_fut)
    # c2_fut = SkyCoord(ra = RA_fut,dec =DEC_fut)
    # sep_fut = [max(c2_fut[c_mem].separation(c2_fut)) for c_mem in range(len(c2_fut))]
    # rad_fut = max(sep_fut)/2
    res_RA = cen_RA - cen_RA_fut
    res_DEC = cen_DEC - cen_DEC_fut 
    
    
    
    ax[1].scatter(RA_fut + res_RA, DEC_fut + res_DEC, label = '%.2f Myr foward'%(ff.value/1e6))
    ax[1].scatter(cen_RA,cen_DEC, s = 200, color ='r')
    # ax[1].scatter(cen_RA_fut,cen_DEC_fut, s = 200, color ='black')
    ax[1].invert_xaxis()
    ax[1].legend()
    
    # clus_dbs[:,0], clus_dbs[:,1]= RA_fut + res_RA, DEC_fut + res_DEC
    np.savetxt(cls_to_throw + '%s_%.2fMyr.txt'%(choosen_cluster,ff.value/1e6), 
               np.c_[(RA_fut + res_RA).value,(DEC_fut + res_DEC).value,clus_dbs[:,2:6]], fmt = (2*'%.8f ' + 4*' %.4f'), header = 'RA, DEC, pmra, pmdec, f127, f153,')
# %%
# est = 309
# v = np.sqrt(clus_dbs[est,2]**2 + clus_dbs[est,3]**2)
# ang = np.degrees(np.arctan2(clus_dbs[est,3],clus_dbs[est,2]))

# v_ = np.sqrt(x[est]**2 + y[est]**2)
# ang_ = np.degrees(np.arctan2(y[est],x[est]))

# print('%.2f, %.2f'%(v,v_))
# print('%.2f, %.2f'%(ang,ang_))

# %%
#

# ang2 = np.degrees(np.arctan2(y, x))

# print(y/x, np.sqrt(x**2 + y**2))
# # print(DEC_cl/ RA_cl)

# for i in range(len(ang)):
#     print('This is diff = %.10f'%(ang[i].to(u.mas).value-ang2[i].to(u.mas).value))
















