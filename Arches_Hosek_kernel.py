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
# =============================================================================
# #Choose Arches or Quintuplet
# =============================================================================
choosen_cluster = 'Arches'
# choosen_cluster = 'Quintuplet'

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
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
# clustered_by = 'all_color'
clustered_by = 'all'

    

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
# SECOND distance plot
# here he are going to select a smaller subgroup of the data and check if the 
# new kernel method of selecting epsilon we will get the same cluster, aprox
# 
# =============================================================================



# %%
# =============================================================================
# Selecting reduced data
# =============================================================================
# Now that we can find a cluster, we are going to tryint again changing the distance, e.g. zooming in the data
# so, we choose randomnly a cluster point and performn the clustering only on the points within a certain distance
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

rand_clus = clus_gal[rand]
rand_pm = pm_clus[rand]
radio=2*u.arcsec

#Here we can decide if selected the reduced data set around a random value of the cluster.

# =============================================================================
# id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(rand_clus,arc_gal, radio)
# dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(rand_clus,clus_gal, radio)
# 
# =============================================================================
# or around just a random point in the data set, that can be part of the found cluster or not
# This one is not ready jet
# =============================================================================
# id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(rand_all,arc_gal, radio)
# dbs_clus, id_arc_dbs, d2d_db, d3d_db = ap_coor.search_around_sky(rand_clus,clus_gal, radio)
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




fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[1].set_title('THIS ONE,Radio = %s, Orange = %s'%(radio,len(dbs_clus)))
# ax[0].set_title('%s, method: %s'%(choosen_cluster,method))
ax[0].set_title('%s, std(mu_l,mu_b): %.3f, %.3f'%(choosen_cluster,
                                           np.std(pm_clus.pm_l_cosb[id_arc_dbs].value),
                                           np.std(pm_clus.pm_b[id_arc_dbs].value)))
plotting('l','b',arc_gal.l, arc_gal.b,1,alpha=0.01,color='k')
plotting('l','b',clus_gal.l[id_arc_dbs], clus_gal.b[id_arc_dbs],1,color='orange',alpha=0.3,zorder=3)


plotting('mul','mub',pm_gal.pm_l_cosb, pm_gal.pm_b,0,alpha=0.3)
plotting('mul','mub',pm_clus.pm_l_cosb[id_arc_dbs], pm_clus.pm_b[id_arc_dbs],0,alpha=0.8)


diff_color = max(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)-min(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)
ax[2].set_title('diff color = %.3f, std_color=%.3f'%(diff_color,np.std(m127_clus[id_arc_dbs].value-m153_clus[id_arc_dbs].value)))
plotting('m127-m153','m153',m127-m153, m153,2,zorder=1,alpha=0.1)
plotting('m127-m153','m153',m127_clus[id_arc_dbs]-m153_clus[id_arc_dbs],m153_clus[id_arc_dbs],2,alpha=0.8)
ax[2].invert_yaxis()


ax[0].invert_xaxis()

sys.exit('line 417')

# %%
# =============================================================================
# DBSCAN in reduced area
# =============================================================================
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

area_l,area_b = arc_gal.l[id_arc],arc_gal.b[id_arc]
area_pml,area_pmb = pml[id_arc], pmb[id_arc]
area_m153,area_m127 = m153[id_arc],m127[id_arc]
area_colorines = colorines[id_arc]
samples_dist_area = 10
if clustered_by =='all_color':
    X_area=np.array([area_pml,area_pmb,area_l,area_b,area_colorines]).T 
    X_stad_area =  StandardScaler().fit_transform(X_area)
    
    tree_area=KDTree(X_stad_area, leaf_size=2) 
    dist_area, ind_area = tree_area.query(X_stad_area, k=samples_dist_area) 
    d_KNN_area=sorted(dist_area[:,-1])#distance to the Kth neighbour

elif clustered_by == 'all':
    X_area = np.array([area_pml,area_pmb]).T
    X_stad_area =  StandardScaler().fit_transform(X_area)
    
    tree_area=KDTree(X_stad_area, leaf_size=2) 
    dist_area, ind_area = tree.query(X_stad_area, k=samples_dist_area) 
    d_KNN_area=sorted(dist_area[:,-1])#distance to the Kth neighbour

#here we generate the kernel simulated data 
area_pml_kernel, area_pmb_kernel = gaussian_kde(area_pml), gaussian_kde(area_pmb)
area_l_kernel, area_b_kernel = gaussian_kde(area_l), gaussian_kde(area_b)
area_color_kernel = gaussian_kde(area_colorines)

lst_d_KNN_sim_area = []
for d in range(20):
    mub_sim_area,  mul_sim_area =area_pmb_kernel.resample(len(area_pml)), area_pml_kernel.resample(len(area_pml))
    l_sim_area, b_sim_area = area_l_kernel.resample(len(area_pml)), area_b_kernel.resample(len(area_pml))
    color_sim_area = color_kernel.resample(len(area_pml))
    if clustered_by == 'all_color':
        X_sim_area=np.array([mul_sim_area[0],mub_sim_area[0],l_sim_area[0],b_sim_area[0],color_sim_area[0]]).T
        X_stad_sim_area = StandardScaler().fit_transform(X_sim_area)
        tree_sim_area =  KDTree(X_stad_sim_area, leaf_size=2)
        
        dist_sim_area, ind_sim_area = tree_sim_area.query(X_stad_sim_area, k=samples_dist_area) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim_area=sorted(dist_sim_area[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim_area.append(min(d_KNN_sim_area))
    elif clustered_by =='all':
        X_sim_area=np.array([mul_sim_area[0],mub_sim_area[0],l_sim_area[0],b_sim_area[0]]).T
        X_stad_sim_area = StandardScaler().fit_transform(X_sim_area)
        tree_sim_area =  KDTree(X_stad_sim_area, leaf_size=2)
        
        dist_sim_area, ind_sim_area = tree_sim_area.query(X_stad_sim_area, k=samples_dist_area) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim_area=sorted(dist_sim_area[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim_area.append(min(d_KNN_sim_area))

d_KNN_sim_av_area = np.mean(lst_d_KNN_sim_area)

fig, ax = plt.subplots(1,1,figsize=(10,10))
# ax[0].set_title('Sub_sec_%s_%s'%(col[colum],row[ro]))
# ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
# ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')

# # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
# ax[0].set_xlabel('Point') 
# ax[0].set_ylabel('%s-NN distance'%(samples)) 
ax.set_title('Number of points = %s '%(len(area_pml)))
ax.hist(d_KNN_area,bins ='auto',histtype ='step',color = 'k')
ax.hist(d_KNN_sim_area,bins ='auto',histtype ='step',color = 'r')
ax.set_xlabel('%s-NN distance'%(samples_dist_area)) 

eps_av_area = round((min(d_KNN_area)+d_KNN_sim_av_area)/2,3)
texto = '\n'.join(('min real d_KNN_area = %s'%(round(min(d_KNN_area),3)),
                    'min sim d_KNN_area =%s'%(round(d_KNN_sim_av_area,3)),'average = %s'%(eps_av_area)))


props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

ax.set_ylabel('N') 



epsilon = eps_av_area
clustering_area = DBSCAN(eps=epsilon, min_samples=samples_dist_area).fit(X_stad_area)
# =============================================================================
# Here ser dbscan manually till you get the same fucking cluster you got using the whole set
# =============================================================================
# clustering_area = DBSCAN(eps=epsilon_area, min_samples=samples_dist_area).fit(X_stad_area)


l_area = clustering_area.labels_

loop_area = 0

n_clusters_area = len(set(l_area)) - (1 if -1 in l_area else 0)
n_noise_area=list(l_area).count(-1)

u_labels_area = set(l_area)
colors_area=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l_area)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


for k in range(len(colors_area)): #give noise color black with opacity 0.1
    if list(u_labels_area)[k] == -1:
        colors_area[k]=[0,0,0,0.1]
        
colores_index_area=[]      
for c in u_labels_area:
    cl_color_area=np.where(l_area==c)
    colores_index_area.append(cl_color_area)


fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[0].invert_xaxis()
# %
elements_in_cluster_area=[]
for i in range(len(set(l_area))-1):
    elements_in_cluster_area.append(len(area_pml[colores_index_area[i]]))
    plotting('mul','mub',area_pml[colores_index_area[i]], area_pmb[colores_index_area[i]],0, color=colors_area[i],zorder=3,alpha=0.1)
    plotting('l','b',area_l[colores_index_area[i]], area_b[colores_index_area[i]],1, color=colors_area[i],zorder=3,alpha=0.3)
    plotting('m127-m153','m153',area_colorines[colores_index_area[i]],area_m153[colores_index_area[i]],2)
    print(len(pml[colores_index_area[i]]))
plotting('m127-m153','m153',m127-m153, m153,2,zorder=1,alpha=0.01)
ax[1].set_title('%s. Larger clsuter = %s '%(choosen_cluster, max(elements_in_cluster_area)))
plotting('mul','mub',pml[colores_index[-1]], pmb[colores_index[-1]],0, color=colors[-1],zorder=1)
# plotting_h('mul','mub',X[:,0][colores_index[-1]], X[:,1][colores_index[-1]],0, color=colors[-1],zorder=1)
plotting('l','b',l[colores_index[-1]], b[colores_index[-1]],1, color=colors[-1],zorder=1,alpha=0.01)
ax[2].invert_yaxis()

# %%





