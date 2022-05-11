#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:05:18 2022

@author: amartinez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""+
Created on Thu Feb 24 09:23:55 2022

@author: amartinez
"""

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
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
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
# We upload galactic center stars, that we will use in the CMD
# catal=np.loadtxt(results+'refined_%s_PM.txt'%(name))
# catal_df=pd.read_csv(pruebas+'%s_refined_with_GNS_partner_mag_K_H.txt'%(name),sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'])

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
dmu_lim = 0.5
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))




# catal=catal_df.to_numpy()
# valid=np.where(np.isnan(catal[:,14])==False)
# mul_mc,mub_mc,dmul_mc,dmub_mc
# gal_coor=catal[:,[17,18,19,20]]


# %%


#mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
Ms_all=np.loadtxt(pruebas +'pm_of_Ms_in_%s.txt'%(name))# this are the information (pm, coordinates and ID) for the Ms that remain in the data after triming it 
group_lst=Ms_all[:,-1]#indtinfication number for the Ms

# pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
pms=[0,0,0,0]
# pms=[0,0,-5.60,-0.20] #this is from the dynesty adjustment
# pms=np.array(pms)

# %%
Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()#TODO
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
AKs_center =AKs_np[center]#TODO
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree)
# %
# %
AKs_list =  np.arange(1.6,2.11,0.01)
# %%
for file_to_remove in glob.glob(pruebas+'dbs_%scluster*.txt'%(pre)):#Remove the files for previpus runs adn radios
    os.remove(file_to_remove) 
cluster_by='all'# this varible can be 'pm' or 'pos', indicating if you want cluster by velocities or positions,or all for clustering in 4D
pixel = 'yes'# yes if you want coordenates in pixels for clustering and plotting positions, insteat of sky coordenates

# print(''*len('ACTIVATE ERASE FILES')+'\n'+'ACTIVATE ERASE FILES'+'\n'+''*len('ACTIVATE ERASE FILES'))
# for g in range(len(group_lst)):
for g in range(0,10):
    seed(g)
    fig, ax = plt.subplots(1,1,figsize=(30,10))
    ax.set_ylim(0,10)
    ax.text(0.0, 5, 'Group %s %s'%(int(group_lst[g]),pre),fontsize= 400,color=plt.cm.rainbow(random()))
    seed(g)
    ax.text(0.5, 2, '\n'+'pixel=%s'%(pixel),fontsize= 200,color=plt.cm.rainbow(random()))

    # print(group_lst[g])
    samples=10# number of minimun objects that defined a cluster
    samples_dist = samples# the distance to the kth neightbour that will define the frist epsilon for debsacn to star looping
    group=int(group_lst[g])
    #ra,dec,x_c,y_c,mua,dmua,mud,dmud,time,n1,n2,idt,m139,Separation,Ks,H,mul,mub,l,b
    # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
    # r_u=[22,32,43,76]#this are the radios around the MS
    r_u=[76]#this are the radios around the MS

    for r in  range(len(r_u)):
        # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'"
        data=np.loadtxt(pruebas + 'group_radio%s_%s_%s.txt'%(r_u[r],group,name))
        
        this=np.where(Ms_all[:,-1]==group)
        Ms=Ms_all[this]
    # %  
        ra_=data[:,5]
        dec_=data[:,6]
        # Process needed for the trasnformation to galactic coordinates
        c = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree, frame='fk5')#you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames
        gal_c=c.galactic
        
        t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))  
        
    # %
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
        
        # if cluster_by == 'pm':
        #     X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3]]).T #Select pm (galactic)
        # elif cluster_by == 'pm_color':
        #     X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],data[:,3]-data[:,4]]).T #Select pm (galactic)
        # elif cluster_by == 'pos':
        #     # X=np.array([t_gal['l'].value,t_gal['b'].value]).T #Select position (galactic)
        #     X=np.array([data[:,7],data[:,8]]).T #Select position (pixels)
        # elif cluster_by == 'all':
        #     X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],t_gal['l'].value,t_gal['b'].value]).T# in Castro-Ginard et al. 2018 they cluster the data in a 5D space: pm,position and paralax    
        # elif cluster_by == 'all_pixel':
        #     X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],data[:,7],data[:,8]]).T# using pixel position instead of coordinates for the clustering   
        # elif cluster_by == 'all_color':
        #     X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],t_gal['l'].value,t_gal['b'].value,data[:,3]-data[:,4]]).T#
        # X_stad = StandardScaler().fit_transform(X)
        # print('These are the mean and std of X: %s %s'%(round(np.mean(X_stad),1),round(np.std(X_stad),1)))
        #THis is how I do it 
        tree=KDTree(X_stad, leaf_size=2) 
    
        
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
        # d_KNN=sorted(dist[:,1])# this is how Ban do it
    
        # This how Ban do it
        # nn = NearestNeighbors(n_neighbors=samples, algorithm ='kd_tree')
        # nn.fit(X_stad)# our training is basically our dataset itself
        # dist, ind = nn.kneighbors(X_stad,samples)
        # d_KNN = np.sort(dist, axis=0)
        # d_KNN = d_KNN[:,1] # this is the difference in bans method. She is selecting the distance to the closest 1st neigh. I choose the last one d_KNN[:,-1]
        
        # eps_for_mean=[np.mean(dist[i]) for i in range(len(dist))]
        # eps_for_mean =[]
        # for i in range(len(dist)):
         
        #     eps_for_mean.append(np.mean(dist[i]))
        
        kneedle = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
        elbow = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
        rodilla=round(kneedle.elbow_y, 3)
        codo = round(elbow.elbow_y, 3)
        
        
        
    
       
    
    
    
    
        # % tutorial at https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        
        # epsilon=np.mean(eps_for_mean)
        # epsilon=codo
        epsilon = round(min(d_KNN),3)
        # sys.exit('salida')
        # epsilon=0.5575
        clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X_stad)
        l=clustering.labels_
        loop=0
        while len(set(l))<3: # min number of cluster to find. It star looking at the min values of the Knn distance plot and increases epsilon until the cluster are found. BE careful cose ALL cluster will be found with the lastest (and biggest) value of eps, so it might lost some clusters, becouse of the conditions.
                             # What I mean is that with a small epsilon it may found a cluster that fulfill the condition (max diff of color), but when increasing epsilon some other stars maybe added to the cluster with a bigger diff in color and break the rule.
                             # This does not seem a problem when 'while <6' but it is when 'while <20' for example...
            loop +=1
            clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X_stad)
            
            l=clustering.labels_
            epsilon +=0.001 # if choose epsilon as min d_KNN you loop over epsilon and a "<" simbol goes in the while loop
            # samples +=1 # if you choose epsilon as codo, you loop over the number of sambles and a ">" goes in the  while loop
            print('DBSCAN loop %s. Trying with eps=%s. cluster = %s '%(loop,round(epsilon,3),len(set(l))-1))
            if loop >100:
                print('breaking out')
                break
               
        # print('breaking the loop')
        print('This is the number of clusters: %s'%(len(set(l))-1))
        
        # =============================================================================
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot(np.arange(0,len(data),1),d_KNN)
        # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
        ax.set_xlabel('Point') 
        ax.set_ylabel('%s-NN distance'%(samples)) 
        # print(round(kneedle.knee, 3))
        # print(round(kneedle.elbow, 3))
        # print(round(kneedle.knee_y, 3))
        # print(round(kneedle.elbow_y, 3))
        ax.axhline(rodilla,linestyle='dashed',color='k')
        ax.axhline(codo,linestyle='dashed',color='k')
        ax.axhline(round(min(d_KNN),3),linestyle='dashed',color='k')
        ax.axhline(epsilon,linestyle='dashed',color='red') 
        ax.text(len(X)/2,epsilon, '%s'%(round(epsilon,3)),color='red')

        ax.text(0,codo, '%s'%(codo))
        ax.text(0,rodilla, '%s'%(rodilla))
        ax.fill_between(np.arange(0,len(X)), codo, rodilla, alpha=0.5, color='grey')
        # =============================================================================
        
        # %Plots the vector poits plots for all the selected stars
    # =============================================================================
    #     fig, ax = plt.subplots(1,1,figsize=(8,8))
    #     # ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)
    #     ax.scatter(data[:,-4],data[:,-3],s=10,alpha=0.5)
    #     # ax.set_xlim(-15,15)
    #     # ax.set_ylim(-15,15)
    #     ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
    #     ax.set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
    #     ax.set_title('Group %s'%(group))
    # =============================================================================
        #%
        
        
        
        n_clusters = len(set(l)) - (1 if -1 in l else 0)
        print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
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
        # print(colores_index)
        if n_clusters > 0:
            #This plots the need plot
    # =============================================================================
    #         fig, ax = plt.subplots(1,1,figsize=(8,8))
    #         ax.plot(np.arange(0,len(data),1),d_KNN)
    #         ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
    #         ax.set_xlabel('Point') 
    #         ax.set_ylabel('%s-NN distance'%(samples)) 
    #         # print(round(kneedle.knee, 3))
    #         # print(round(kneedle.elbow, 3))
    #         # print(round(kneedle.knee_y, 3))
    #         # print(round(kneedle.elbow_y, 3))
    #         ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k')
    # =============================================================================
            
            # print(''*len('RESTORE CONDITIONS')+'\n'+'RESTORE CONDITIONS'+'\n'+''*len('RESTORE CONDITIONS'))

            for i in range(len(set(l))-1):
                
                min_c=min(data[:,3][colores_index[i]]-data[:,4][colores_index[i]])
                max_c=max(data[:,3][colores_index[i]]-data[:,4][colores_index[i]])
                min_Ks=min(data[:,4][colores_index[i]])
                min_nth = np.sort(data[:,4][colores_index[i]])
                index1=np.where((catal[:,5]==Ms[0,4]) & (catal[:,6]==Ms[0,5]) ) # looping a picking the stars coord on the Ms catalog
                
                min_c_J=min(data[:,2][colores_index[i]]-data[:,4][colores_index[i]])
                max_c_J=max(data[:,2][colores_index[i]]-data[:,4][colores_index[i]])
                


                # if max_c-min_c <0.3 and (len(min_nth))>3 and min_nth[2]<14.5:# the difference in color is smaller than 'max_c-min_c' and at least min_nth[n] stars in the cluster are brighter than 14.5
                if max_c-min_c <0.5 and any(min_nth<14.5):# the difference in color is smaller than 'max_c-min_c' and at least one star in the cluster are brighter than 14.5
                    # index1=np.where((catal[:,5]==Ms[0,4]) & (catal[:,6]==Ms[0,5]) ) # looping a picking the stars coord on the Ms catalog
                    print(Ms[0,4],Ms[0,5])
                    print(catal[:,5][index1],catal[:,6][index1])
                    print(index1)
                    # index=np.where((catal_all[:,0]==yso_ra[i]) & (catal_all[:,1]==yso_dec[i]) ) # this finding the MS in the whole libralati data, that is not trimmed, so its contains all the MS (well 96 of then the rest are in the other Libralato catalog)
               
                    
                    fig, ax = plt.subplots(1,4,figsize=(40,10))
                    # fig, ax = plt.subplots(1,3,figsize=(30,10))
                    ax[2].invert_yaxis()
                    seed(g)
                    ax[0].set_title('Group %s, radio = %s, # of Clusters = %s'%(group,r_u[r], n_clusters),color=plt.cm.rainbow(random()))
                    seed(g)
                    ax[1].set_title('# of stars = #%s, eps=%s'%(len(l),round(epsilon,3)),color=plt.cm.rainbow(random()))
                    # t_gal['l'] = t_gal['l'].wrap_at('180d')
                    ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
                    ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
                    # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])

                    ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)
                    ax[0].set_xlim(-10,10)
                    ax[0].set_ylim(-10,10)
                    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
                    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
                
                    ax[0].scatter(Ms[0,0]-pms[2],Ms[0,1]-pms[3],s=50,color='red',marker='2',zorder=3)
                    ax[0].scatter(pms[2],pms[3],s=150, marker='*')
                    ax[0].invert_xaxis()
                    # Here we save the coordenates of the posible cluster coordinates for further anlysis if required
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
                    #ra,dec,x_c,y_c,mua,dmua,mud,dmud,time,n1,n2,idt,m139,Separation,Ks,H,mul,mub,l,b
                    #data="'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','dmul','dmub','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
                    
                    # ax[1].scatter(data[:,0][colores_index[i]],data[:,1][colores_index[i]], color=colors[i],s=50)#plots in ecuatorials
                    if pixel =='no':
                        t_gal['l'] = t_gal['l'].wrap_at('180d')
                        
                       
                        ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
                        ax[1].scatter(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, color=colors[i],s=50,zorder=3)#plots in galactic
                        ax[1].scatter(t_gal['l'].value,t_gal['b'].value, color=colors[-1],s=50,zorder=1,alpha=0.5)#plots in galactic
                        ax[1].quiver(t_gal['l'].value,t_gal['b'].value, X[:,0]-pms[2], X[:,1]-pms[3], alpha=0.5, color=colors[-1])
                        ax[1].scatter(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, color=colors[i],s=50,zorder=3)#plots in galactic           
                        ax[1].quiver(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, X[:,0][colores_index[i]]-pms[2], X[:,1][colores_index[i]]-pms[3], alpha=0.5, color=colors[i])
                        ax[1].set_xlabel('l') 
                        ax[1].set_ylabel('b') 
                        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                        if len(index1[0]) > 0:
                            # mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
                            Ms_co = SkyCoord(ra = Ms[0,4]*u.deg, dec = Ms[0,5]*u.deg, frame ='icrs').galactic
                            ax[1].scatter(Ms_co.l.wrap_at('180d'), Ms_co.b,s=50, color='red', marker='2')
                        else:
                            Ms_co = SkyCoord(ra = Ms[0,4]*u.deg, dec = Ms[0,5]*u.deg, frame ='icrs').galactic
                            ax[1].scatter(Ms_co.l.wrap_at('180d'), Ms_co.b,color='red',s=50,marker='o', facecolors='none', edgecolors='r')
                    else:
                        ax[1].scatter(X[:,2], X[:,3], color=colors[-1],s=50,zorder=3)#plots in galactic
                        ax[1].quiver(X[:,2], X[:,3], X[:,0]-pms[2], X[:,1]-pms[3], alpha=0.5, color=colors[-1],zorder=1)
                        ax[1].quiver(X[:,2][colores_index[-1]], X[:,3][colores_index[-1]], X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
                        ax[1].scatter(X[:,2][colores_index[i]], X[:,3][colores_index[i]], color=colors[i],s=50,zorder=3)#plots in galactic
                        ax[1].quiver(X[:,2][colores_index[i]], X[:,3][colores_index[i]], X[:,0][colores_index[i]]-pms[2], X[:,1][colores_index[i]]-pms[3], alpha=0.5, color=colors[i])
                        ax[1].set_xlabel('x') 
                        ax[1].set_ylabel('y') 
                        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                        if len(index1[0]) > 0:
                            # mul, mub, mua, mud, ra, dec,dmul,dmub,x,y position in GALCEN_TABLE_D.cat 
                            ax[1].scatter(Ms[0,-3], Ms[0,-2],s=50, color='red', marker='2',zorder=3)
                        else:
                            ax[1].scatter(Ms[0,-3], Ms[0,-2],color='red',s=50,marker='o', facecolors='none', edgecolors='r')
                    
            #Ms=mul, mub, mua, mud, ra, dec,dmul,dmub,l,b,Ks, H, m139, position in GALCEN_TABLE_D.cat 
        
                    # ax[1].scatter(Ms[0,8],Ms[0,9],s=100,color='red',marker='2')
                    # ax[1].quiver(data[:,17][colores_index[i]], data[:,18][colores_index[i]], X[:,0][colores_index[i]], X[:,1][colores_index[i]], alpha=0.5, color=colors[i])#galactic
            
                    # ax[1].set_xlabel('ra') 
                    # ax[1].set_ylabel('dec') 
                    
                    
                    
                    
                    if pixel == 'no':
                        clus_array = np.array([data[:,5][colores_index[i]],data[:,6][colores_index[i]],t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value,
                                                                                              X[:,0][colores_index[i]], 
                                                                                              X[:,1][colores_index[i]],
                                                                                              data[:,3][colores_index[i]],data[:,4][colores_index[i]]]).T
                        clus_array1= np.c_[clus_array, np.full((len(X[:,0][colores_index[i]]),1),g),np.full((len(X[:,0][colores_index[i]]),1),i)]
                        np.savetxt(pruebas + 'dbs_%scluster%s_of_group%s.txt'%(pre,i,g),clus_array1,fmt='%.7f '*8 + '%.0f '*2, header ='ra, dec, l, b, pml, pmb, H, Ks, group, cluster')
                    elif pixel == 'yes':
                        clus_array = np.array([data[:,5][colores_index[i]],data[:,6][colores_index[i]],data[:,7][colores_index[i]], data[:,8][colores_index[i]],
                                                                                              X[:,0][colores_index[i]], 
                                                                                              X[:,1][colores_index[i]],
                                                                                              data[:,3][colores_index[i]],data[:,4][colores_index[i]]]).T
                        clus_array1= np.c_[clus_array, np.full((len(X[:,0][colores_index[i]]),1),g),np.full((len(X[:,0][colores_index[i]]),1),i)]
                        np.savetxt(pruebas + 'dbs_%scluster%s_of_group%s.txt'%(pre,i,g),clus_array1,fmt='%.7f '*8 + '%.0f '*2, header ='ra, dec, x, y, pml, pmb, H, Ks, group, cluster')
                    
                    # %'ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'
                    # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
                
                    
                    seed(g)
                    ax[2].set_title('#%s/112,min stars/cluster = %s, cluster#=%s'%(g, samples,i),color=plt.cm.rainbow(random()))
# =============================================================================
#                     # First we identify the member of the cluster in the extintion catalog
# =============================================================================
                    clus_coord =  SkyCoord(ra=data[:,5][colores_index[i]]*u.degree, dec=data[:,6][colores_index[i]]*u.degree)
                    idx = clus_coord.match_to_catalog_sky(gns_coord)
                    gns_match = AKs_center[idx[0]]
                    good = np.where(gns_match[:,11] == -1)
                    if len(good[0]) != len(gns_match[:,11]):
                        print('%s foreground stars in this cluster'%(len(gns_match[:,11]) - len(good)))
                    gns_match_good = gns_match[good]
                    AKs_clus_all = [float(gns_match_good[i,18]) for i in range(len(gns_match_good[:,18]))  if gns_match_good[i,18] !='-']
                    # print('\n'.join((40*'@','%s'%(np.mean(AKs_clus_all)), '%s'%(np.std(AKs_clus_all)),40*'@')))
# =============================================================================
#                     Second, we build the synthetic cluster
# =============================================================================
                    AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                    absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                    
                    AKs = min(AKs_list, key=absolute_difference_function)
                    frase = 'Diff in extintion bigger than 0.2!'
                    print('\n'.join((10*'§','%s %s'%(AKs_clus,AKs),10*'§')))
                    if abs(AKs - AKs_clus)>0.2:
                        print(''*len(frase)+'\n'+frase+'\n'+''*len(frase))
                        
                    iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
                    
                    dist = 8000 # distance in parsec
                    metallicity = 0.17 # Metallicity in [M/H]
                    # logAge = np.log10(0.61*10**9.)
                    logAge = np.log10(0.010*10**9.)
                    
                    evo_model = evolution.MISTv1() 
                    atm_func = atmospheres.get_merged_atmosphere
                    red_law = reddening.RedLawNoguerasLara18()
                    filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
                    
                    iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                                    evo_model=evo_model, atm_func=atm_func,
                                                    red_law=red_law, filters=filt_list,
                                                        iso_dir=iso_dir)
                    
                    
                    #%
                    
                    
                    imf_multi = multiplicity.MultiplicityUnresolved()
                    
                    # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01
                    
                    # NOTE: when defining the power law slope for each segment of the IMF, we define
                    # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
                    # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 
                    
                    massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
                    powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
                    # my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
                    my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)
                    
                    
                    #%
                    
                    
                    mass = 0.5*10**4.
                    mass = 1 * mass
                    dAks = round(std_AKs*1,3)
                    cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
                    cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
                    clus = cluster.star_systems
                    clus_ndiff = cluster_ndiff.star_systems
                    
                    ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'r',label='With dAKs = %s mag'%(dAks))
                    ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',label='With dAKs = %s mag'%(0),alpha=0.3)
                    ax[2].legend(loc =3, fontsize = 12)
                    
                    # p.arrow(max_c+max_c/5,min_Ks+0.5,-(max_c+max_c/5-max_c),0,head_width=0.05,color=colors[i])
                    ax[2].scatter(data[:,3][colores_index[i]]-data[:,4][colores_index[i]],data[:,4][colores_index[i]], color=colors[i],s=50,zorder=2)
                    if len(index1[0]) > 0:
                            ax[2].scatter((catal[:,3][index1]-catal[:,4][index1]),catal[:,4][index1], color='springgreen',s=100,marker='2',zorder=3)
                    ax[2].axvline(min_c,color=colors[i],ls='dashed',alpha=0.5)
                    ax[2].axvline(max_c,color=colors[i],ls='dashed',alpha=0.5)
                    ax[2].annotate('%s'%(round(max_c-min_c,3)),(max_c+max_c/5,min_Ks+0.5),color=colors[i])
                    
                    # ax[2].scatter((Ms[0,11]-Ms[0,10]),Ms[0,10], color='red',s=100,marker='2',zorder=3)
                    # ax[2].set_xlim(0,)
                    ax[2].set_xlim(1.0,2.2)
                    ax[2].set_xlabel('H$-$Ks') 
                    ax[2].set_ylabel('Ks') 
                    txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
                                         'age = %.0f Myr'%(10**logAge/10**6)))
                    txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    # place a text box in upper left in axes coords
                    ax[2].text(0.05, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    ax[2].text(0.05, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    if len(index1[0]) > 0:
                            ax[2].scatter((catal[:,3][index1]-catal[:,4][index1]),catal[:,4][index1], color='springgreen',s=400,marker='2',zorder=3)
# =============================================================================
#  # %                #This is for J-Ks plot.
# =============================================================================
 
                    ax[3].scatter(clus['m_hawki_J']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'r',label='With dAKs = %s mag'%(dAks))
                    ax[3].scatter(clus_ndiff['m_hawki_J']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',label='With dAKs = %s mag'%(0),alpha=0.3)
                    ax[3].legend(loc =3, fontsize = 12)
                    
                    ax[3].scatter(data[:,2][colores_index[i]]-data[:,4][colores_index[i]],data[:,4][colores_index[i]], color=colors[i],s=50,zorder=2)
                    if len(index1[0]) > 0:
                            ax[3].scatter((catal[:,2][index1]-catal[:,4][index1]),catal[:,4][index1], color='springgreen',s=100,marker='2',zorder=3)
                    ax[3].axvline(min_c_J,color=colors[i],ls='dashed',alpha=0.5)
                    ax[3].axvline(max_c_J,color=colors[i],ls='dashed',alpha=0.5)
                    ax[3].annotate('%s'%(round(max_c_J-min_c_J,3)),(max_c_J+max_c_J/5,min_Ks+0.5),color=colors[i])
                    
                    
                    # ax[3].set_xlim(1.0,2.2)
                    ax[3].set_xlabel('J$-$Ks') 
                    ax[3].set_ylabel('Ks') 
                    txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
                                         'age = %.0f Myr'%(10**logAge/10**6)))
                    txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    # place a text box in upper left in axes coords
                    ax[3].text(0.75, 0.95, txt_AKs, transform=ax[3].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    ax[3].text(0.75, 0.85, txt_srn, transform=ax[3].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    if len(index1[0]) > 0:
                            ax[3].scatter((catal[:,2][index1]-catal[:,4][index1]),catal[:,4][index1], color='springgreen',s=400,marker='2',zorder=3)
                    ax[3].invert_yaxis()            
            
# %%
print('$\sigma_{mul}$')






