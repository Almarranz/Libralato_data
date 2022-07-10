#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:24:02 2022

@author: amartinez
"""
# Appling the kernel method to Arches data. The point is to have a justification of its usage
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
# Only data with valid color and uncertainties in pm smaller than 0.4
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


# sys.exit('line 67')
# %%
columnas=str(arches.columns)
arc_coor=SkyCoord(ra=arches['ra*']*u.arcsec+center_arc.ra,dec=arches['dec']*u.arcsec+ center_arc.dec)
# %%
ra, dec =arc_coor.ra, arc_coor.dec
e_ra,e_dec = arches['e_ra*']*u.arcsec, arches['e_dec']*u.arcsec
# %%
pmra, pmdec = arches['pm_ra*']*u.mas/u.yr, arches['pm_dec']*u.mas/u.yr
e_pmra, e_pmdec = arches['e_pm_ra*'].value, arches['e_pm_dec'].value
print(np.std(e_pmra),np.std(e_pmdec))
# %%
m127, m153 = arches['F127M']*u.mag,arches['F153M']*u.mag

# =============================================================================
# np.savetxt(pruebas + 'arches_for_topcat.txt',np.array([ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value]).T,header='ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value')
# =============================================================================
# %%
arc_gal=arc_coor.galactic
pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic


l,b=arc_gal.l, arc_gal.b
pml,pmb=pm_gal.pm_l_cosb, pm_gal.pm_b
colorines = m127-m153
# %Thi is for the extinction

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
# clus_coord =  SkyCoord(ra=datos[:,5][colores_index[i][0]]*u.degree, dec=datos[:,6][colores_index[i][0]]*u.degree)
# idx = clus_coord.match_to_catalog_sky(gns_coord)

# %

# %% Definition section
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl
# %
def plotting_h(namex,namey,x,y,ind,**kwargs):
    try:
        pl=ax[ind].hexbin(x.value,y.value,**kwargs)
    except:
        pl=ax[ind].hexbin(x,y,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl



# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting_h('l','b',l,b,0,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('mul','mub',pml,pmb,1,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('m127-m153','m153',m127-m153,m153,2,norm=matplotlib.colors.LogNorm())

#
# =============================================================================
# Generated part
# =============================================================================
clustered_by = 'all_color'#TODO
# clustered_by = 'all'#TODO

    

# %

# X_stad=X

samples_dist=7


#here we generate the kernel simulated data 
pml_kernel, pmb_kernel = gaussian_kde(pml), gaussian_kde(pmb)
l_kernel, b_kernel = gaussian_kde(l), gaussian_kde(b)
color_kernel = gaussian_kde(colorines)

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
mul_kernel, mub_kernel = gaussian_kde(pml), gaussian_kde(pmb)
l_kernel, b_kernel = gaussian_kde(l), gaussian_kde(b)
color_kernel = gaussian_kde(colorines)

lst_d_KNN_sim = []
for d in range(20):
    mub_sim,  mul_sim = mub_kernel.resample(len(pmb)), mul_kernel.resample(len(pml))
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
ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

ax.set_ylabel('N') 


# =============================================================================
# DBSCAN part
# =============================================================================
epsilon = eps_av
clustering = DBSCAN(eps = epsilon, min_samples=samples_dist).fit(X_stad)

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
elements_in_cluster=[]
for i in range(len(set(l_c))-1):
    elements_in_cluster.append(len(pml[colores_index[i]]))
    plotting('mul','mub',pml[colores_index[i]], pmb[colores_index[i]],0, color=colors[i],zorder=3)
    plotting('l','b',l[colores_index[i]], b[colores_index[i]],1, color=colors[i],zorder=3)
    plotting('m127-m153','m153',m127[colores_index[i]]-m153[colores_index[i]],m153[colores_index[i]],2,color=colors[i], zorder=3)
    print(len(pml[colores_index[i]]))
ax[0].set_title('n of cluster = %s,eps=%s,min size=%s'%(n_clusters,round(epsilon,2),samples_dist))
ax[1].set_title('%s. Larger cluster = %s'%(choosen_cluster, max(elements_in_cluster)))
plotting('mul','mub',pml[colores_index[-1]], pmb[colores_index[-1]],0, color=colors[-1],zorder=1)
# plotting_h('mul','mub',X[:,0][colores_index[-1]], X[:,1][colores_index[-1]],0, color=colors[-1],zorder=1)
plotting('l','b',l[colores_index[-1]], b[colores_index[-1]],1, color=colors[-1],zorder=1)
ax[2].invert_yaxis()
plotting('m127-m153','m153',m127[colores_index[-1]]-m153[colores_index[-1]],m153[colores_index[-1]],2,color=colors[-1],zorder=1)




# %%
# =============================================================================
# Selecting reduced data
# =============================================================================
def plotting_h(namex,namey,x,y,ind,**kwargs):
    try:
        pl=ax[ind].hexbin(x.value,y.value,**kwargs)
    except:
        pl=ax[ind].hexbin(x,y,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl

clus_gal=arc_gal[colores_index[0]]
pm_clus=pm_gal[colores_index[0]]
m153_clus = m153[colores_index[0]]
m127_clus = m127[colores_index[0]]
arches_small = arches[colores_index[0]]
# =============================================================================
# # NOte to myself: pm_clus is a Skycoord pm obeject
# # , that is not the same than a Skycoor coord objet. 
# # The former stores coord and pm and, aparently to acces 
# # the proper motions coordinate you have to do it separetly
# # , i.e. pm_clus.pm_l_cosb or pm_clu.b(doing pm_clus.pm does not work)
# =============================================================================
# pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic






rand = np.random.choice(np.arange(0,len(clus_gal)),1)
rand_all = np.random.choice(np.arange(0,len(pml)),1)




radio=2*u.arcsec#TODO

#Here we can decide if selected the reduced data set around a random value of the cluster.

# =============================================================================
# rand_clus = clus_gal[rand]
# id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(rand_clus,arc_gal, radio)
# dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(rand_clus,clus_gal, radio)
# 
# =============================================================================
# or around just a random point in the data set, that can be part of the found cluster or not
# This one is not ready jet

# =============================================================================
# rand_pm = pm_clus[rand]
# id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(rand_all,arc_gal, radio)
# dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(rand_clus,clus_gal, radio)
# 
# =============================================================================

# or around the pre-dertermined coordenates for center of the cluster
# frist one selects stars whithin a distance 'radio' around the centere of the cluster
# secondo one selects stara whithin a distance 'radio' arpind the center of the cluster, but only for the star labeled as cluster stars
# Notice that for small radios around the center both values are very similar.
id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(SkyCoord(['17h45m50.4769267s'], ['-28d49m19.16770s'], frame='icrs'),arc_gal, radio) if choosen_cluster =='Arches' else ap_coor.search_around_sky(SkyCoord(['17h46m15.13s'], ['-28d49m34.7s'], frame='icrs'),arc_gal, radio)
dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(SkyCoord(['17h45m50.4769267s'], ['-28d49m19.16770s'], frame='icrs'),clus_gal, radio) if choosen_cluster =='Arches' else ap_coor.search_around_sky(SkyCoord(['17h46m15.13s'], ['-28d49m34.7s'], frame='icrs'),clus_gal, radio)

#search_around_sky complains when one of the variable is just a singe coordinates (and not an array of coordinates)
#so in order to go around this put the coordinares in brackets and it woint complain any more

# %
fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[1].set_title('THAT ONE, Radio = %s, Green = %s'%(radio,len(id_clus)))
ax[0].set_title('%s,std(mu_l,mu_b): %.3f, %.3f'%(choosen_cluster,np.std(pml[id_arc].value),np.std(pmb[id_arc].value)))
plotting('l','b',arc_gal.l, arc_gal.b,1)
plotting('l','b',clus_gal.l, clus_gal.b,1,color='orange')
plotting('l','b',arc_gal.l[id_arc], arc_gal.b[id_arc],1,alpha=0.9,color='g')


plotting('mul','mub',pm_gal.pm_l_cosb, pm_gal.pm_b,0)
plotting('mul','mub',pm_clus.pm_l_cosb, pm_clus.pm_b,0)
plotting('mul','mub',pml[id_arc], pmb[id_arc],0,alpha=0.1)
ax[0].invert_xaxis()


plotting('m127-m153','m153',m127-m153, m153,2,zorder=1,alpha=0.01)
plotting('m127-m153','m153',m127_clus-m153_clus, m153_clus,2,alpha=0.3,color='orange')
plotting('m127-m153','m153',m127[id_arc]-m153[id_arc],m153[id_arc],2,alpha=0.8,color='g')
# This line was only for checking I was selecting the right stars
# plotting('m127-m153','m153',arches_small['F127M']-arches_small['F153M'],arches_small['F153M'],2,color='r',s=1)



# %
fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[1].set_title('THIS ONE,Radio = %s, Orange = %s'%(radio,len(dbs_clus)))
# ax[0].set_title('%s, method: %s'%(choosen_cluster,method))
ax[0].set_title('%s, std(mu_l,mu_b): %.3f, %.3f'%(choosen_cluster,
                                           np.std(pm_clus.pm_l_cosb[id_arc_dbs].value),
                                           np.std(pm_clus.pm_b[id_arc_dbs].value)))
ax[1].scatter(arc_gal.l, arc_gal.b,alpha=0.01,color='k')
ax[1].scatter(clus_gal.l[id_arc_dbs], clus_gal.b[id_arc_dbs],color='orange',alpha=0.3,zorder=3)
ax[1].set_xlabel('l(deg)',fontsize =30) 
ax[1].set_ylabel('b(deg)',fontsize =30) 

ax[0].scatter(pm_gal.pm_l_cosb, pm_gal.pm_b,alpha=0.3)
ax[0].scatter(pm_clus.pm_l_cosb[id_arc_dbs], pm_clus.pm_b[id_arc_dbs],alpha=0.8)
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 


diff_color = max(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)-min(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)
ax[2].set_title('diff color = %.3f, std_color=%.3f'%(diff_color,np.std(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)))
plotting('m127-m153','m153',m127-m153, m153,2,zorder=1,alpha=0.1)
plotting('m127-m153','m153',m127_clus[id_arc_dbs]-m153_clus[id_arc_dbs],m153_clus[id_arc_dbs],2,alpha=0.8,s=100)

good_pro = np.where(arches_small[id_arc_dbs]['Pclust']>0.60)#TODO
print(len((arches_small['Pclust'][id_arc_dbs][good_pro])))
plotting('m127-m153','m153',arches_small['F127M'][id_arc_dbs][good_pro]-arches_small['F153M'][id_arc_dbs][good_pro],
         arches_small['F153M'][id_arc_dbs][good_pro],2,color = 'r',s=80,marker='x')
ax[2].invert_yaxis()

ax[2].invert_yaxis()


ax[0].invert_xaxis()

# =============================================================================
# sys.exit('\n'.join(('The scripts stops at the line 424, and does not run dbscan again','We are just selecting the star at the core of the cluster.',
#                     'The idea is to have a model of a how a small cluster would look like at the NSD',
#                     'Check the velocities and the CMD', 
#                    'Use this core sample to calculate the mass with spysea')))
# 
# =============================================================================
# %

# %
# =============================================================================
# Here we are going to plot the core cluster, figure out the mass and fit an isochrone.
# Since Spisea does not have the filter used by Libralato et al, we have to match the whole catalog 
# in order to be able to plot the cluster and also the backgroun stars
# =============================================================================
color_de_cluster = 'lime'

fig, ax = plt.subplots(1,3, figsize=(30,10))
prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
ax[1].text(0.10, 0.95, 'aprox cluster radio = %s" \n cluster stars = %s '%(radio,len(id_arc_dbs)), transform=ax[1].transAxes, fontsize=30,verticalalignment='top', bbox=prop)

mul_mean_all, mub_mean_all = np.mean(pm_gal.pm_l_cosb),np.mean(pm_gal.pm_b)
mul_mean, mub_mean = np.mean(pm_clus.pm_l_cosb[id_arc_dbs]),np.mean(pm_clus.pm_b[id_arc_dbs])

mul_sig_all, mub_sig_all = np.std(pm_gal.pm_l_cosb),np.std(pm_gal.pm_b)
mul_sig, mub_sig = np.std(pm_clus.pm_l_cosb[id_arc_dbs]),np.std(pm_clus.pm_b[id_arc_dbs])

vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean.value,3), round(mub_mean.value,3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig.value,3), round(mub_sig.value,3)))) 
vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all.value,3), round(mub_mean_all.value,3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all.value,3), round(mub_sig_all.value,3))))

propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.3)
propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
    verticalalignment='top', bbox=propiedades)
ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
    verticalalignment='top', bbox=propiedades_all)





ax[0].scatter(pm_gal.pm_l_cosb, pm_gal.pm_b,color = 'k',alpha = 0.03)
ax[0].scatter(pm_clus.pm_l_cosb[id_arc_dbs], pm_clus.pm_b[id_arc_dbs],color = color_de_cluster,alpha = 1)
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
ax[0].invert_xaxis()



prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
ax[1].text(0.10, 0.95, 'aprox cluster radio = %s" \n cluster stars = %s '%(radio,len(id_arc_dbs)), transform=ax[1].transAxes, fontsize=30,verticalalignment='top', bbox=prop)
ax[1].scatter(arc_gal.l.value, arc_gal.b.value,alpha=0.01,color='k')
ax[1].scatter(clus_gal[id_arc_dbs].l,clus_gal[id_arc_dbs].b,alpha=1,color=color_de_cluster)


ax[1].set_xlabel('l(deg)',fontsize =30) 
ax[1].set_ylabel('b(deg)',fontsize =30) 

# clus_coord =  SkyCoord(ra=datos[:,5][colores_index[i][0]]*u.degree, dec=datos[:,6][colores_index[i][0]]*u.degree)
# idx = clus_coord.match_to_catalog_sky(gns_coord)

ax[2].scatter(m127-m153, m153,alpha=0.01,color='k')
ax[2].scatter(m127_clus[id_arc_dbs]-m153_clus[id_arc_dbs],m153_clus[id_arc_dbs],alpha=1,color=color_de_cluster)
ax[2].set_xlabel('f127m-f153m',fontsize =30) 
ax[2].set_ylabel('f153m',fontsize =30) 
ax[2].invert_yaxis()

# %


print(arches.columns)

















gns_coord_gal = gns_coord.galactic
# this returns 3 values: 0-> the index, 1-> 2d distance, 2->3s distanc3(unuseful in this case)
common_stars = clus_gal[id_arc_dbs].match_to_catalog_sky(gns_coord_gal)
AKs_cluster = AKs_center[common_stars[0]] 
is_match = np.where(common_stars[1]<1*u.arcsec)
gns_match = gns_coord_gal[common_stars[0][is_match]]
AKs_cluster = AKs_cluster[is_match]
print(common_stars)



ax[1].scatter(gns_match.l,gns_match.b,alpha=1,color='r',s = 1)
# ax[1].set_xlim(0.122,0.125)
# ax[1].set_ylim(0.015,0.020)
# %
# =============================================================================
# Here we create the isochrone with spisea
# =============================================================================

ext_cluster = []
for ext in range(len(AKs_cluster)):
    ext_cluster.append(float(AKs_cluster[ext,18]))
# print(np.mean(ext_cluster),np.std(ext_cluster))

iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'

dist = 8000 # distance in parsec
metallicity = 0.30 # Metallicity in [M/H]
# # logAge_600 = np.log10(0.61*10**9.)
if choosen_cluster =='Arches':
    logAge = np.log10(0.0025*10**9.)#TODO
elif choosen_cluster == 'Quintuplet':
    logAge = np.log10(0.0048*10**9.)

evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()
filt_list = ['wfc3,ir,f127m', 'wfc3,ir,f153m']

iso_dir = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/iso_dir/'
# print(iso.points.columns)
# names=('L','Teff','R','mass','logg','isWR','mass_current','phase','m_hst_f127m','m_hst_f153m')>
iso =  synthetic.IsochronePhot(logAge, np.mean(ext_cluster), dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

imf_multi = multiplicity.MultiplicityUnresolved()

# # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01

# # NOTE: when defining the power law slope for each segment of the IMF, we define
# # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
# # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 

massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
# my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)

ax[2].plot(iso.points['m_hst_f127m'] - iso.points['m_hst_f153m'], 
                    iso.points['m_hst_f153m'], 'b-',  label='%.2f Myr'%(10**logAge/1e6),alpha =0.5)
ax[2].legend()
ax[2].set_ylim(max(m153_clus[id_arc_dbs].value),min(m153_clus[id_arc_dbs].value)-0.5)

M_mass = 1*10**4.
# mass = 1 * mass
dAks = round(np.std(ext_cluster),3)
cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, M_mass,0.05)
# cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
clus = cluster.star_systems
# clus_ndiff = cluster_ndiff.star_systems
ax[2].scatter(clus['m_hst_f127m']-clus['m_hst_f153m'],clus['m_hst_f153m'],color = 'r',alpha=0.2)

props = dict(boxstyle='round', facecolor='k', alpha=0.3)
txt_AKs = '\n'.join(('AKs = %.2f'%(np.mean(ext_cluster)),
                     'std_AKs = %.2f'%(np.std(ext_cluster))))
ax[2].text(0.65, 0.50, txt_AKs, transform=ax[2].transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

# %%
mag_127, mag_153 = m127_clus[id_arc_dbs], m153_clus[id_arc_dbs]
max_stars = len(mag_127)*2
porcentaje = 0.0
M_mass = 1*10**4.
loop =0
while  max_stars > len(mag_127)+0.3*len(mag_127):
    
    # mass = 0.8*10**4.
    mass = M_mass - 0.01*porcentaje*M_mass
    # dAks = std_AKs[0]
    dAks = 0.05
    loop += 1
    print(loop)
    cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
    cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
    clus = cluster.star_systems
    clus_ndiff = cluster_ndiff.star_systems
    
    max_mass = np.where((clus_ndiff['m_hst_f153m']>min(mag_153.value))&(clus_ndiff['m_hst_f153m']<max(mag_153.value)))
    
    max_stars = len(clus_ndiff['m_hst_f153m'][max_mass])
    porcentaje +=1

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].hist(clus['mass'],bins = 'auto',color ='k')#, label ='Cluster Mass = %.0f$M_{\odot}$ \n virial mass = %.0f'%(mass,M_clus.value) )
ax[0].set_xlabel('$(M_{\odot})$')
ax[0].set_ylabel('$N$')
   
   
# ax[1].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'slategray',alpha=0.7)
ax[1].scatter( m127-m153, m153,alpha=0.01,color='k',s=50)
ax[1].invert_yaxis()
ax[1].scatter(clus_ndiff['m_hst_f127m']-clus_ndiff['m_hst_f153m'],clus_ndiff['m_hst_f153m'],color =color_de_cluster,s=100)
props = dict(boxstyle='round', facecolor='w', alpha=0.5)

ax[1].text(0.55, 0.95, 'L mass = %.0f $M_{\odot}$'%(mass), transform=ax[1].transAxes, fontsize=25,
    verticalalignment='top', bbox=props)
ax[1].set_xlabel('f127m-f153m')
ax[1].set_ylabel('f153m')
plt.show()

# %%
print(mag_153)
    
    
    
    
    
    
    
    
    
