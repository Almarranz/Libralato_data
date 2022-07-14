#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:57:10 2022

@author: amartinez
"""

# =============================================================================
# Looks for the cluster in Hoseck`s data and then select the very core of it and 
# then matches the cluster with gns survey and using spisea plots a simulated cluster.
# =============================================================================
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.coordinates import SkyCoord
import astropy.coordinates as ap_coor
import astropy.units as u
from matplotlib import rcParams
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from sklearn.preprocessing import StandardScaler
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import pandas as pd
from astropy.table import Column
from astropy.coordinates import FK5
from astropy.stats import sigma_clip

# %%

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

#%% 
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pruebas='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

# =============================================================================
# #Choose Arches or Quintuplet
# =============================================================================
choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits') if choosen_cluster =='Arches' else Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')

# %% Here we are going to trimm the data
# Only data with valid color and uncertainties in pm smaller than 0.4 and excluding foreground stars by color-cut
m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]
arches=arches[valid_colors]

center = np.where((m127.value - m153.value > 1.7) &(m127.value - m153.value < 4))
arches = arches[center]

epm_gal = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec, pm_ra_cosdec =  arches['e_pm_ra*']*u.mas/u.yr, pm_dec = arches['e_pm_dec']*u.mas/u.yr,frame = 'icrs').galactic
pme_lim = 0.4
valid_epm = np.where((epm_gal.pm_l_cosb.value < pme_lim)&(epm_gal.pm_b.value < pme_lim))
arches=arches[valid_epm]

# %%
hos_coord = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec)
hos_gal = SkyCoord(ra = hos_coord.ra, dec = hos_coord.dec, frame = 'icrs').galactic
pmra, pmdec = arches['pm_ra*']*u.mas/u.yr, arches['pm_dec']*u.mas/u.yr

columnas = len(arches.columns)
if columnas < 26:
    arches.add_column(hos_coord.ra,name='ra_abs',index=0)
    arches.add_column(hos_coord.dec,name='dec_abs',index=1)
    arches.add_column(hos_gal.l,name='l_abs',index=2)
    arches.add_column(hos_gal.b,name='b_abs',index=3)
    pm_gal = SkyCoord(ra  = arches['ra_abs'] ,dec = arches['dec_abs'], pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic
    pml, pmb = pm_gal.pm_l_cosb, pm_gal.pm_b
    arches.add_column(pml.value,name='pm_l',index=4)
    arches.add_column(pmb.value,name='pm_b',index=5)
elif columnas == 26:
    print('ra and dec already added to Hoseck data: \n',arches.columns)
# %%
print(arches.columns)
# %%


#%%
clustered_by = 'all_color'#TODO
# clustered_by = 'all'#TODO
samples_dist=7
# %%
#here we generate the kernel simulated data 

colorines = arches['F127M']-arches['F153M']
pml_kernel, pmb_kernel = gaussian_kde(arches['pm_l']), gaussian_kde(arches['pm_b'])
l_kernel, b_kernel = gaussian_kde(arches['l_abs'].value), gaussian_kde(arches['b_abs'].value)
color_kernel = gaussian_kde(colorines)
# %%
pml, pmb = arches['pm_l'],arches['pm_b']
l,b = arches['l_abs'].value,  arches['b_abs'].value
if clustered_by == 'all_color':
    X = np.array([pml,pmb,l,b,colorines]).T
    X_stad = StandardScaler().fit_transform(X)
    tree = KDTree(X_stad, leaf_size=2) 
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
elif clustered_by == 'all':
    X = np.array([pml,pmb,l,b]).T
    X_stad = StandardScaler().fit_transform(X)
    tree = KDTree(X_stad, leaf_size=2) 
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour


lst_d_KNN_sim = []
for d in range(20):
    mub_sim,  mul_sim = pmb_kernel.resample(len(pmb)), pml_kernel.resample(len(pml))
    l_sim, b_sim = l_kernel.resample(len(pml)), b_kernel.resample(len(pmb))
    color_sim = color_kernel.resample(len(pml))
    if clustered_by == 'all_color':
        X_sim=np.array([mul_sim[0],mub_sim[0],l_sim[0],b_sim[0],color_sim[0]]).T
        X_stad_sim = StandardScaler().fit_transform(X_sim)
        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
        
        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim.append(min(d_KNN_sim))
    elif clustered_by =='all':
        X_sim=np.array([mul_sim[0],mub_sim[0],l_sim[0],b_sim[0]]).T
        X_stad_sim = StandardScaler().fit_transform(X_sim)
        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
        
        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim.append(min(d_KNN_sim))

d_KNN_sim_av = np.mean(lst_d_KNN_sim)


fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('Number of points = %s '%(len(pml)))

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
ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

ax.set_ylabel('N') 
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


for k in range(len(colors)): #give noise color black with opacity 0.1
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
        
colores_index=[]      
for c in u_labels:
    cl_color=np.where(l_c==c)
    colores_index.append(cl_color)
    
fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[0].invert_xaxis()
ax[2].invert_yaxis()
elements_in_cluster=[]
for i in range(len(set(l_c))-1):
    ax[0].scatter(pml[colores_index[i]], pmb[colores_index[i]],color=colors[i],zorder=3)
    # ax[1].scatter(l[colores_index[i]], b[colores_index[i]],color=colors[i],zorder=3)
    ax[1].scatter(arches[colores_index[i]]['ra_abs'],arches[colores_index[i]]['dec_abs'],color=colors[i],zorder=3,s=100)
    # ax[1].scatter(gns_match[colores_index[i]][:,0],gns_match[colores_index[i]][:,2],color=colors[i],zorder=3,s=100)
    ax[2].scatter(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]],arches['F153M'][colores_index[i]],color=colors[i],zorder=13)
    
ax[0].scatter(pml[colores_index[-1]], pmb[colores_index[-1]],color=colors[-1],zorder=1)
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
# ax[1].scatter(l[colores_index[-1]], b[colores_index[-1]],color=colors[-1],zorder=1)
ax[1].scatter(arches[colores_index[-1]]['ra_abs'],arches[colores_index[-1]]['dec_abs'],color=colors[-1],zorder=3,s=100,alpha = 0.01)
ax[1].set_xlabel('ra(deg)',fontsize =30) 
ax[1].set_ylabel('dec(deg)',fontsize =30)

ax[2].scatter(arches['F127M'][colores_index[-1]]-arches['F153M'][colores_index[-1]],arches['F153M'][colores_index[-1]],color=colors[-1],zorder=1)
ax[2].set_xlabel('f127m-f153m',fontsize =30) 
ax[2].set_ylabel('f153m',fontsize =30) 
# %%
hos_cluster = arches[colores_index[0]]
print(hos_cluster.columns)
# %%
# %%
# =============================================================================
# Here we are going to match with GNS
# =============================================================================
Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()
center = np.where((AKs_np[:,6]-AKs_np[:,8] > 1.3) &(AKs_np[:,6]<90)&(AKs_np[:,8]<90))
AKs_center =AKs_np[center]
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree,frame = FK5,equinox ='J2000')


# %% I cosider a math if the stars are less than 1 arcsec away 

print(hos_cluster['ra_abs'])
# %%
# =============================================================================
# Here we are going to select the core of the cluster buy selecting the DBSCAN
# cluster scan around a distance equal to 'radio' of the center of the Arches or
# quintuplet cluster
# =============================================================================

radio=3*u.arcsec#TODO
hos_coor_clus = SkyCoord(ra = hos_cluster['ra_abs'], dec = hos_cluster['dec_abs'],frame = 'icrs')
id1, id2, d2d,d3d = ap_coor.search_around_sky(SkyCoord(['17h45m50.4769267s'], ['-28d49m19.16770s'], frame='icrs'),hos_coor_clus, radio) if choosen_cluster =='Arches' else ap_coor.search_around_sky(SkyCoord(['17h46m15.13s'], ['-28d49m34.7s'], frame='icrs'),hos_coor_clus, radio)
# dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(SkyCoord(['17h45m50.4769267s'], ['-28d49m19.16770s'], frame='icrs'),clus_gal, radio) if choosen_cluster =='Arches' else ap_coor.search_around_sky(SkyCoord(['17h46m15.13s'], ['-28d49m34.7s'], frame='icrs'),clus_gal, radio)

fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.scatter(arches['ra_abs'],arches['dec_abs'],color ='k',alpha = 0.03)
ax.scatter(hos_cluster[id2]['ra_abs'],hos_cluster[id2]['dec_abs'],color = 'g')

hos_core = hos_cluster[id2]

hos_core_coord = SkyCoord(ra = hos_core['ra_abs'], dec = hos_core['dec_abs'],frame ='icrs')
idi, d2d, d3d = hos_core_coord.match_to_catalog_sky(gns_coord)
is_match = np.where(d2d<1*u.arcsec)

hos_core_match=hos_core[is_match]
gns_core_match = AKs_center[idi[is_match]]

#%%
# =============================================================================
# Here you can trim the core cluster by color
# =============================================================================
colores_trim = []
sig = 2
for tr in range(len(gns_core_match)):
    colores_trim.append(float(gns_core_match[tr,6]-gns_core_match[tr,8]))
print(np.mean(colores_trim))
fil_color = sigma_clip(colores_trim, sigma=sig, maxiters=10)
good_filt = np.where(np.isnan(fil_color)==False)

gns_core_match_trim = gns_core_match[good_filt]

# %


#Checking the matching, you can delete these tree lines
fig, ax = plt.subplots(1,3, figsize = (30,10))
ax[0].scatter(arches['pm_l'],arches['pm_b'],color = 'k', alpha = 0.03)
ax[0].scatter(hos_core_match['pm_l'],hos_core_match['pm_b'],color = 'lime')
vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(np.mean(hos_core_match['pm_l']),3), round(np.mean(hos_core_match['pm_b']),3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(np.std(hos_core_match['pm_l']),3), round(np.std(hos_core_match['pm_b']),3)))) 
vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(np.mean(arches['pm_l']),3), round(np.mean(arches['pm_b']),3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(np.std(arches['pm_l']),3), round(np.std(arches['pm_b']),3))))

propiedades = dict(boxstyle='round', facecolor='lime' , alpha=0.3)
propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
    verticalalignment='top', bbox=propiedades)
ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
    verticalalignment='top', bbox=propiedades_all)


ax[1].set_title('%s'%(choosen_cluster))
ax[1].scatter(hos_core_match['ra_abs'], hos_core_match['dec_abs'])
ax[1].scatter(gns_core_match_trim[:,0],gns_core_match_trim[:,2],color = 'r',s=1)
ax[1].scatter(arches['ra_abs'],arches['dec_abs'],color ='k',alpha = 0.03)

prop_1 = dict(boxstyle='round', facecolor='lime' , alpha=0.2)
ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(radio.to(u.arcsec).value,2),len(gns_core_match_trim)), transform=ax[1].transAxes, fontsize=30,
                                            verticalalignment='top', bbox=prop_1)
                    

# ax[2].scatter(hos_core_match['F127M']-hos_core_match['F153M'], hos_core_match['F153M'],zorder =3)
# ax[2].scatter(arches['F127M']-arches['F153M'], arches['F153M'],zorder=1)
ax[2].set_title('Stars trimmied by color at %s$\sigma$'%(sig))
ax[2].scatter(AKs_center[:,6]-AKs_center[:,8],AKs_center[:,8],zorder =1, color = 'k',s=0.1,alpha = 0.01)
ax[2].scatter(gns_core_match_trim[:,6]-gns_core_match_trim[:,8],gns_core_match_trim[:,6],zorder =3, color = 'lime')
ax[2].set_xlim(1.2,4)
ax[2].invert_yaxis()
# %


ext_cluster = []
for ext in range(len(gns_core_match_trim[:,18])):
    ext_cluster.append(float(gns_core_match_trim[ext,18]))


AKs_core, dAKs_core = np.median(ext_cluster), np.std(ext_cluster)


# %
iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'

AKs_list1 =  np.arange(1.6,2.11,0.01)
AKs_list = np.append(AKs_list1,0)#I added the 0 for the isochrones without extiction
absolute_difference_function = lambda list_value : abs(list_value - AKs_core)
AKs = min(AKs_list, key=absolute_difference_function)

dist = 8200 # distance in parsec
metallicity = 0.30 # Metallicity in [M/H]
# # logAge_600 = np.log10(0.61*10**9.)
if choosen_cluster =='Arches':
    logAge = np.log10(0.0025*10**9.)#TODO
elif choosen_cluster == 'Quintuplet':
    logAge = np.log10(0.0048*10**9.)

evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()
filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']

iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                               evo_model=evo_model, atm_func=atm_func,
                               red_law=red_law, filters=filt_list,
                               iso_dir=iso_dir)

imf_multi = multiplicity.MultiplicityUnresolved()


massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
# my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)

ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
                                      iso.points['m_hawki_Ks'], 'b-',   label='%.2f Myr'%(10**logAge/1e6),alpha =0.5)
ax[2].legend()
mass = 1*10**4.
mass = 1 * mass
dAks = round(dAKs_core,3)
# dAks = 0.08
cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
clus = cluster.star_systems
clus_ndiff = cluster_ndiff.star_systems

props = dict(boxstyle='round', facecolor='k', alpha=0.3)
txt_AKs = '\n'.join(('AKs = %.2f'%(np.mean(ext_cluster)),
                     'std_AKs = %.2f'%(np.std(ext_cluster))))
ax[2].text(0.65, 0.50, txt_AKs, transform=ax[2].transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'r',alpha=0.1)
ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.1,s=1)


# %%


























