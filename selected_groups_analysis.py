#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:07:29 2022

@author: amartinez
"""
# %% imports
import astropy.coordinates as ap_coor
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
# %%Plotting
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
# %%
# Here we are going to have a close up of the group with a putative cluster in it.
# So far these groups are selected by eye, after inspection of the outcome from dbsacan_comparation.py
name='WFC3IR'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
# name='ACSWFC'
pixel =  'yes'
eje_x_pos = 'x (pixel)' if pixel =='yes' else 'l'
eje_y_pos = 'y (pixel)' if pixel =='yes' else 'b'
cluster_by = 'all'
trimmed_data = 'yes'
if trimmed_data == 'yes':
    pre=''
elif trimmed_data == 'no':
    pre='relaxed_'
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))

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
 # dmu_lim = 0.5
dmu_lim = 0.5
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]


gal_coord=SkyCoord(ra = catal[:,5]*u.deg, dec = catal[:,6]*u.deg, frame = 'icrs').galactic
# %%
clus_id = 1
search = 'dbs'
groups = [65]# groups to be analyzed
radio = 76*u.arcsec # radio(s) of the list you will like to explore
cluster_by = 'all'# or vel or pos
pms=[0,0,0,0]# if you want to substract the value of the movenment of SgrA* give values to pms[pm_ra,pm_dec,pm_l,pm_b]
# for each gruop up to six cluster could be found. Manually decide which one of them you want to inspec foward


for g in groups:
    
    # ' ra, dec, x, y, pml, pmb, H, Ks'
    cluster = np.loadtxt(pruebas + '%s_%scluster%s_of_group%s.txt'%(search, pre,clus_id,g))
    ra_ = cluster[:,0]
    dec_ = cluster[:,1]
    print(ra_[0],dec_[0])
    # Process needed for the trasnformation to galactic coordinates
    gal_c = SkyCoord(ra = cluster[:,0]*u.degree, dec = cluster[:,1]*u.degree, frame='icrs').galactic#you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames
    # gal_c=c.galactic
    
    t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))  
    
    index = np.where((catal[:,5] == ra_[0]) & (catal[:,6] == dec_[0]))
    
    id_clus, id_catal, d2d,d3d = ap_coor.search_around_sky(gal_coord[index],gal_coord, radio)

    def plotting(namex,namey,x,y,ind,**kwargs):
        
        pl=ax[ind].scatter(x,y,**kwargs)
        
        try:
            ax[ind].set_xlabel('%s (%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
            ax[ind].set_ylabel('%s (%s)'%(namey, y.unit))
        except:
            ax[ind].set_xlabel('%s'%(namex)) 
            ax[ind].set_ylabel('%s'%(namey))
        if ind ==2:
            ax[ind].invert_yaxis()
        return pl
# %
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    # plotting('l','b',gal_coord[id_catal].l,gal_coord[id_catal].b,1)
    # plotting('l','b',gal_c.l,gal_c.b,1)
    min_c=min(cluster[:,-4]-cluster[:,-3])
    max_c=max(cluster[:,-4]-cluster[:,-3])
    min_Ks=min(cluster[:,-3])
    ax[0].set_title('previous cluster',color = 'r')
    ax[2].set_title('Group %s cluster %s'%(g,clus_id))
    plotting('x','y',catal[:,7][id_catal],catal[:,8][id_catal],1)
    plotting('x','y',cluster[:,2],cluster[:,3],1)
    plotting('mul (mas/yr)','mub (mas/yr)',catal[:,17][id_catal],catal[:,18][id_catal],0)
    plotting('mul (mas/yr)','mub (mas/yr)',cluster[:,4],cluster[:,5],0)
    plotting('H-Ks','Ks',catal[id_catal][:,3]-catal[id_catal][:,4],catal[id_catal][:,4],2)
    plotting('H-Ks','Ks',cluster[:,-4]-cluster[:,-3],cluster[:,-3],2)
    # ax[2].axvline(min_c,color='r',ls='dashed',alpha=0.5)
    # ax[2].axvline(max_c,color='r',ls='dashed',alpha=0.5)
    # ax[2].annotate('%s'%(round(max_c-min_c,3)),(max_c+max_c/5,min_Ks+0.5),color='r')
    ax[0].invert_xaxis()
    ax[2].invert_yaxis()
    # %
    # pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
    pms=[0,0,0,0]           
    # pms=[0,0,-5.60,-0.20] #this is from the dynesty adjustment
    # pms=np.array(pms)
    
    data = catal[id_catal]#selecteing the stars around the putative cluster
    if pixel == 'no':
        X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],t_gal['l'].value,t_gal['b'].value]).T
    elif pixel == 'yes':
        X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],data[:,7],data[:,8]]).T
        
    if cluster_by == 'pos':
        X_stad = StandardScaler().fit_transform(X[:,[2,3]])
    elif cluster_by == 'pm':
        X_stad = StandardScaler().fit_transform(X[:,[0,1]])
    elif cluster_by == 'all':
        X_stad = StandardScaler().fit_transform(X)
    
    samples_dist = 6
    tree=KDTree(X_stad, leaf_size=2)     
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    
    kneedle = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
    elbow = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
    rodilla=round(kneedle.elbow_y, 3)
    codo = round(elbow.elbow_y, 3)
    # epsilon = round(min(d_KNN),3)
    epsilon = 0.4
    
    clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)
    l=clustering.labels_
# =============================================================================
#     loop=0
#     while len(set(l))<5:
#         loop +=1
#         clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)
#         
#         l=clustering.labels_
#         # epsilon +=0.001 # if choose epsilon as min d_KNN you loop over epsilon and a "<" simbol goes in the while loop
#         samples_dist +=1 # if you choose epsilon as codo, you loop over the number of sambles and a ">" goes in the  while loop
#         print('DBSCAN loop %s. Trying with eps=%s. cluster = %s '%(loop,round(epsilon,3),len(set(l))-1))
#         if loop >100:
#             print('breaking out')
#             break
# =============================================================================
           
    # print('breaking the loop')
    print('This is the number of clusters: %s'%(len(set(l))-1))
    
# =============================================================================
#     fig, ax = plt.subplots(1,1,figsize=(8,8))
#     ax.plot(np.arange(0,len(data),1),d_KNN)
#     # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
#     ax.set_xlabel('Point') 
#     ax.set_ylabel('%s-NN distance'%(samples_dist)) 
#     # print(round(kneedle.knee, 3))
#     # print(round(kneedle.elbow, 3))
#     # print(round(kneedle.knee_y, 3))
#     # print(round(kneedle.elbow_y, 3))
#     ax.axhline(rodilla,linestyle='dashed',color='k')
#     ax.axhline(codo,linestyle='dashed',color='k')
#     ax.axhline(epsilon,linestyle='dashed',color='red') 
#     ax.text(len(X)/2,epsilon, '%s'%(round(epsilon,3)),color='red')
#     
#     ax.text(0,codo, '%s'%(codo))
#     ax.text(0,rodilla, '%s'%(rodilla))
#     ax.fill_between(np.arange(0,len(X)), codo, rodilla, alpha=0.5, color='grey')
# =============================================================================
    
    n_clusters = len(set(l)) - (1 if -1 in l else 0)
    print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(g,round(epsilon,2),samples_dist,n_clusters))
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
    def plotting(namex,namey,x,y,ind,**kwargs):
        
        pl=ax[ind].scatter(x,y,**kwargs)
        
        try:
            ax[ind].set_xlabel('%s (%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
            ax[ind].set_ylabel('%s (%s)'%(namey, y.unit))
        except:
            ax[ind].set_xlabel('%s'%(namex)) 
            ax[ind].set_ylabel('%s'%(namey))
        if ind ==2:
            ax[ind].invert_yaxis()
        return pl
    
     
    colores_index=[]
    
    for c in u_labels:
        cl_color=np.where(l==c)
        colores_index.append(cl_color)
        
        
    
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    ax[1].set_title('epsilon = %s'%(round(epsilon,3)))
    ax[0].set_title('previous cluster',color = 'r')
    ax[2].set_title('Group %s cluster %s'%(g,clus_id))
    plotting('mul (mas/yr)','mub (mas/yr)',cluster[:,4],cluster[:,5],0,color = 'r',s=100,zorder=3)
    plotting('x','y',cluster[:,2],cluster[:,3],1,color = 'r',s=100,zorder=3)
    ax[1].quiver(catal[:,7][id_catal],catal[:,8][id_catal],catal[:,17][id_catal],catal[:,18][id_catal])
    plotting('x','y',catal[:,7][id_catal],catal[:,8][id_catal],1,alpha=0.3)
    
    plotting('mul (mas/yr)','mub (mas/yr)',catal[:,17][id_catal],catal[:,18][id_catal],0,alpha=0.3)
   
    # plotting('H-Ks','Ks',catal[id_catal][:,3]-catal[id_catal][:,4],catal[id_catal][:,4],2,alpha=0.3)
    plotting('H-Ks','Ks',catal[:,3]-catal[:,4],catal[:,4],2,alpha=0.01)
    
    
    plotting('H-Ks','Ks',cluster[:,-4]-cluster[:,-3],cluster[:,-3],2,s=100,color ='r')
    ax[2].axvline(min_c,color='r',ls='dashed',alpha=0.5)
    ax[2].axvline(max_c,color='r',ls='dashed',alpha=0.5)
    ax[2].annotate('%s'%(round(max_c-min_c,3)),(max_c+max_c/5,min_Ks+0.5),color='r')
    ax[0].invert_xaxis()
    
    ax[2].set_xlim(1.1,2.2)
    for c in range(len(set(l))-1):
        plotting('mul (mas/yr)','mub (mas/yr)',X[:,0][colores_index[c]],X[:,1][colores_index[c]],0, color=colors[c],s=50,zorder=3,)
        plotting(eje_x_pos,eje_y_pos,X[:,2][colores_index[c]],X[:,3][colores_index[c]],1, color=colors[c],s=50,zorder=3,)
        plotting('H-Ks','Ks',data[colores_index[c]][:,3] -data[colores_index[c]][:,4],data[colores_index[c]][:,4],2,color=colors[c])
        ax[1].quiver(X[:,2][colores_index[c]],X[:,3][colores_index[c]],X[:,0][colores_index[c]],X[:,1][colores_index[c]],color=colors[c])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
