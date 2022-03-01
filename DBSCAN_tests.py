#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:23:55 2022

@author: amartinez
"""

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import StandardScaler
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
cluster_by='pm'# this varible can be 'pm' or 'pos', indicating if you want cluster by velocities or positions
alpha_g=192.85948
delta_g = 27.12825
tr=np.deg2rad

pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
name='WFC3IR'
#mul, mub, mua, mud, ra, dec,x_c,y_c,l,b position in GALCEN_TABLE_D.cat 
Ms_all=np.loadtxt(pruebas +'pm_of_Ms_in_WFC3IR.txt')# this are the information (pm, coordinates and ID) for the Ms that remain in the data after triming it 
group_lst=Ms_all[:,-1]#indtinfication number for the Ms

# pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
pms=[0,0,0,0]



# for g in range(len(group_lst)):
for g in range(1):
    # print(group_lst[g])
    samples=5# number of minimun objects that defined a cluster
    
    group=int(group_lst[g])
    #ra,dec,x_c,y_c,mua,dmua,mud,dmud,time,n1,n2,idt,m139,Separation,Ks,H,mul,mub,l,b
    data=np.loadtxt(pruebas + 'group_%s_%s.txt'%(group,name))
    
    this=np.where(Ms_all[:,-1]==group)
    Ms=Ms_all[this]
# %%
    if cluster_by == 'pm':
        X=np.array([data[:,-4],data[:,-3]]).T #Select pm (galactic)
    elif cluster_by == 'pos':
        X=np.array([data[:,-2],data[:,-1]]).T #Select position (galactic)
        
    X = StandardScaler().fit_transform(X)
    tree=KDTree(X, leaf_size=2) 

    # dist, ind = tree.query(iris[:,0:2], k=5)
    dist, ind = tree.query(X, k=samples)
    d_KNN=sorted(dist[:,-1])
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.plot(np.arange(0,len(data),1),d_KNN)
    kneedle = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='convex')
    # print(round(kneedle.knee, 3))
    # print(round(kneedle.elbow, 3))
    # print(round(kneedle.knee_y, 3))
    # print(round(kneedle.elbow_y, 3))
    rodilla=round(kneedle.elbow_y, 3)
    ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k')

   




    # %% tutorial at https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    
    
    epsilon=rodilla
    # epsilon=0.2
    
    clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X)
    
    docu=DBSCAN.__doc__
    
    l=clustering.labels_
    # %%Plots the vector poits plots for all the selected stars
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    # ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)
    ax.scatter(data[:,-4],data[:,-3],s=10,alpha=0.5)
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
    ax.set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
    ax.set_title('Group %s'%(group))
    #%%
    
    
    
    n_clusters = len(set(l)) - (1 if -1 in l else 0)
    print('Number of cluster with eps=%s and min_sambles=%s: %s'%(epsilon,samples,n_clusters))
    n_noise=list(l).count(-1)
    # %%
    u_labels = set(l)
    colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
    # %%
    
    # %%
    for k in range(len(colors)):
        if list(u_labels)[k] == -1:
            colors[k]=[0,0,0,0.1]
    # %%       
    colores_index=[]
    
    for c in u_labels:
        cl_color=np.where(l==c)
        colores_index.append(cl_color)
    # %%
    # print(colores_index)
    
# =============================================================================
#     fig, ax = plt.subplots(1,2,figsize=(20,10))
#     ax[0].set_title('Group %s. # of Clusters = %s'%(group, n_clusters))
#     # for i in range(n_clusters):
#     for i in range(len(set(l))):
#         # fig, ax = plt.subplots(1,1,figsize=(10,10))
#         # ax.set_title('Cluster #%s'%(i+1))
#         ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50)
#         ax[0].set_xlim(-10,10)
#         ax[0].set_ylim(-10,10)
#         ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
#         ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
#         ax[0].scatter(Ms[0,0],Ms[0,1],s=50,color='red',marker='2')
#         ax[0].scatter(pms[2],pms[3],s=150, marker='*')
#         ax[0].invert_xaxis()
#         
#         # ax[1].scatter(data[:,0][colores_index[i]],data[:,1][colores_index[i]], color=colors[i],s=50)#plots in ecuatorials
#         ax[1].scatter(data[:,-2][colores_index[i]],data[:,-1][colores_index[i]], color=colors[i],s=50)#plots in galactic
# 
#         ax[1].scatter(Ms[0,8],Ms[0,9],s=100,color='red',marker='2')
#         # ax[1].quiver(data[:,0][colores_index[i]], data[:,1][colores_index[i]], X[:,0][colores_index[i]]-pms[2], X[:,1][colores_index[i]]-pms[3], alpha=0.5, color=colors[i])#ecuatorial
#         ax[1].quiver(data[:,-2][colores_index[i]], data[:,-1][colores_index[i]], X[:,0][colores_index[i]]-pms[2], X[:,1][colores_index[i]]-pms[3], alpha=0.5, color=colors[i])#galactic
# 
#         # ax[1].set_xlabel('ra') 
#         # ax[1].set_ylabel('dec') 
#         
#         ax[1].set_xlabel('l') 
#         ax[1].set_ylabel('b') 
#         
#         ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#         ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#         
# =============================================================================

    
    # %% Only for velocity space
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.set_title('Group %s. # of Clusters = %s'%(group, n_clusters))
    # for i in range(n_clusters):
    for i in range(n_clusters+1):#The plus one makes plots the poiny that are not cluster
        # fig, ax = plt.subplots(1,1,figsize=(10,10))
        # ax.set_title('Cluster #%s'%(i+1))
        ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=10)
        ax.set_xlim(-15,15)
        ax.set_ylim(-15,15)
        ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
        ax.set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
        ax.scatter(Ms[0,0],Ms[0,1],s=10,color='red')
        
   
    
# %%

   
    
    
    
    
    
