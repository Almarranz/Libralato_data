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
from scipy.stats import gaussian_kde
import astropy.coordinates as ap_coor

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
clustered_by='all_color'#TODO
pixel = 'yes'#TODO yes if you want coordenates in pixels for clustering and plotting positions, insteat of sky coordenates
gen_sim = 'kernnel'#TODO
sim_lim = 'mean'#TODO
#mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
Ms_all=np.loadtxt(pruebas +'pm_of_Ms_in_%s.txt'%(name))# this are the information (pm, coordinates and ID) for the Ms that remain in the data after triming it 
group_lst=Ms_all[:,-1]#indtinfication number for the Ms

# pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
pms=[0,0,0,0]
# pms=[0,0,-5.60,-0.20] #this is from the dynesty adjustment
# pms=np.array(pms)

for file_to_remove in glob.glob(pruebas+'dbs_%scluster*.txt'%(pre)):#Remove the files for previpus runs adn radios
    os.remove(file_to_remove) 

# print(''*len('ACTIVATE ERASE FILES')+'\n'+'ACTIVATE ERASE FILES'+'\n'+''*len('ACTIVATE ERASE FILES'))
for g in range(len(group_lst)):
# for g in range(1,3):
    seed(g)
    fig, ax = plt.subplots(1,1,figsize=(30,10))
    ax.set_ylim(0,10)
    ax.text(0.0, 5, 'Group %s %s'%(int(group_lst[g]),pre),fontsize= 400,color=plt.cm.rainbow(random()))
    seed(g)
    ax.text(0.5, 2, '\n'+'pixel=%s'%(pixel),fontsize= 200,color=plt.cm.rainbow(random()))

    # print(group_lst[g])
    samples=5# number of minimun objects that defined a cluster
    samples_dist = samples# the distance to the kth neightbour that will define the frist epsilon for debsacn to star looping
    group=int(group_lst[g])
    #ra,dec,x_c,y_c,mua,dmua,mud,dmud,time,n1,n2,idt,m139,Separation,Ks,H,mul,mub,l,b
    # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
    # r_u=[22,32,43,76]#this are the radios around the MS
    r_u=[151]#this are the radios around the MS

    for r in  range(len(r_u)):
        # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'"
        data=np.loadtxt(pruebas + 'group_radio%s_%s_%s.txt'%(r_u[r],group,name))
        
        this=np.where(Ms_all[:,-1]==group)
        Ms=Ms_all[this]
    # %  
        ra_=data[:,5]
        dec_=data[:,6]
        # Process needed for the trasnformation to galactic coordinates
        coordenadas =  SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')##you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames
        gal_c=coordenadas.galactic
        
        t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))  
        
    # %
        
        # =============================================================================
        #         Here is where the party begins
        # =============================================================================
                
        datos =[]
        datos = data
        
        
        mul,mub = datos[:,-6],datos[:,-5]
        x,y = datos[:,7], datos[:,8]
        colorines = datos[:,3]-datos[:,4]
        H_datos, K_datos = datos[:,3], datos[:,4] 
        
        mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
        x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
        color_kernel = gaussian_kde(colorines)
        if clustered_by == 'all_color':
            X=np.array([mul,mub,datos[:,7],datos[:,8],colorines]).T
            # X_stad = RobustScaler(quantile_range=(25, 75)).fit_transform(X)#TODO
            X_stad = StandardScaler().fit_transform(X)
            tree = KDTree(X_stad, leaf_size=2) 
            # pca = PCA(n_components=5)
            # pca.fit(X_stad)
            # print(pca.explained_variance_ratio_)
            # print(pca.explained_variance_ratio_.sum())
            # X_pca = pca.transform(X_stad)
            # tree = KDTree(X_pca, leaf_size=2) 
            dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
        elif clustered_by == 'all':
            X=np.array([mul,mub,datos[:,7],datos[:,8]]).T
            # X_stad = RobustScaler(quantile_range=(25, 75)).fit_transform(X)#TODO
            X_stad = StandardScaler().fit_transform(X)
            tree = KDTree(X_stad, leaf_size=2) 
            dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
        # For the simulated data we loop a number of times and get the average of the minimun distance
        lst_d_KNN_sim = []
        if gen_sim == 'kernnel':
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
        if gen_sim == 'shuffle':
            for d in range(20):
                randomize = np.arange(len(datos))
                np.random.shuffle(randomize)
                mul_sim,  mub_sim = mul[randomize], mub[randomize]
                x_sim, y_sim = x, y
                ra_sim, dec_sim = ra_, dec_
                random_col = np.arange(len(datos))
                np.random.shuffle(random_col)
                H_sim, K_sim = H_datos[random_col], K_datos[random_col]
                color_sim = H_sim-K_sim
                if clustered_by == 'all_color':
                    X_sim=np.array([mul_sim,mub_sim,x_sim,y_sim,color_sim]).T
                    X_stad_sim = StandardScaler().fit_transform(X_sim)
                    tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                    
                    dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                    d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                    
                    lst_d_KNN_sim.append(min(d_KNN_sim))
                elif clustered_by =='all':
                    X_sim=np.array([mul_sim,mub_sim,x_sim,y_sim]).T
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
        
        if sim_lim == 'mean':
            eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
            valor = d_KNN_sim_av
        elif sim_lim == 'minimun':
            eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2,3)
            valor = min(lst_d_KNN_sim)
        elif sim_lim == 'maximun':
            eps_av = round((min(d_KNN)+max(lst_d_KNN_sim))/2,3)
            valor = min(lst_d_KNN_sim)
        texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                            'min sim d_KNN =%s'%(round(valor,3)),
                            'average = %s'%(eps_av),'%s'%(sim_lim),'%s'%(gen_sim)))
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
        
        ax.set_ylabel('N') 
        # ax.set_xlim(0,1)
       
        plt.show()
        
        
        clus_method = 'dbs'
    
        clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
        l=clustering.labels_
        
        n_clusters = len(set(l)) - (1 if -1 in l else 0)
        # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
        n_noise=list(l).count(-1)
        # %
        u_labels = set(l)
        colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
        for k in range(len(colors)): #give noise color black with opacity 0.1
            if list(u_labels)[k] == -1:
                colors[k]=[0,0,0,0.1]
        # %      
        colores_index=[]
        
        for c in u_labels:
            cl_color=np.where(l==c)
            colores_index.append(cl_color)
       
        for i in range(len(set(l))-1):
            index1=np.where((catal[:,5]==Ms[0,4]) & (catal[:,6]==Ms[0,5]) ) # looping a picking the stars coord on the Ms catalog
            fig, ax = plt.subplots(1,3,figsize=(30,10))
            color_de_cluster = 'lime'
            # fig, ax = plt.subplots(1,3,figsize=(30,10))
            # ax[2].invert_yaxis()
           
            ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
            # t_gal['l'] = t_gal['l'].wrap_at('180d')
            ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
            ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
            # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
    
            ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=color_de_cluster ,s=50,zorder=3)
            ax[0].set_xlim(-10,10)
            ax[0].set_ylim(-10,10)
            ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
            ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
            ax[0].invert_xaxis()
            ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
            
            ax[0].scatter(Ms[0,0],Ms[0,1],s=200,color='fuchsia',zorder=3)
            
            
            mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
            mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
            
            mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
            mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
        
        
            vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                 '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
            vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                 '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
            
            propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
            propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
            ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
                verticalalignment='top', bbox=propiedades)
            ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
                verticalalignment='top', bbox=propiedades_all)
              
            #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
            c2 = SkyCoord(ra = datos[:,0][colores_index[i]]*u.deg,dec = datos[:,1][colores_index[i]]*u.deg,frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
            sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
            rad = max(sep)/2
            
            radio_MS = max(sep)
            
            # This search for all the points around the cluster that are no cluster
            lista = []
            lista =np.zeros([len(c2),3])
            # for c_memb in range(len(c2)):
            #     distancia = list(c2[c_memb].separation(c2))
            #     # print(int(c_memb),int(distancia.index(max(distancia))),max(distancia).value)
            #     # a =int(c_memb)
            #     # b = int(distancia.index(max(distancia)))
            #     lista[c_memb][0:3]= int(c_memb),int(distancia.index(max(distancia))),max(distancia).value
            
            # coord_max_dist = list(lista[:,2]).index(max(lista[:,2]))
   
            # p1 = c2[int(lista[coord_max_dist][0])]
            # p2 = c2[int(lista[coord_max_dist][1])]

            # m_point = SkyCoord(ra = [(p2.ra+p1.ra)/2], dec = [(p2.dec +p1.dec)/2])
            
            m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)],frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
            
            idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*2)
            
            ax[0].scatter(datos[:,-6][group_md],datos[:,-5][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)

            prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
            ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
                                    verticalalignment='top', bbox=prop)
            
            # ax[1].scatter(catal[:,5], catal[:,6], color='k',s=50,zorder=1,alpha=0.01)#
            ax[1].scatter(datos[:,5],datos[:,6],color='k' ,s=50,zorder=1,alpha=0.01)
            ax[1].scatter(datos[:,5][colores_index[i]],datos[:,6][colores_index[i]],color=color_de_cluster ,s=50,zorder=3)
            
            
            
            ax[1].scatter(datos[:,5][group_md],datos[:,6][group_md],s=50,color='r',alpha =0.1,marker ='x')
            ax[1].set_xlabel('Ra(deg)',fontsize =30) 
            ax[1].set_ylabel('Dec(deg)',fontsize =30) 
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[1].set_title('Area = %.2f arcmin2'%(np.pi*r_u[r]**2/3600))
            
            
            # mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
            # Ms_co = SkyCoord(ra = Ms[0,4]*u.deg, dec = Ms[0,5]*u.deg, frame ='icrs')
            # ax[1].scatter(Ms_co.l.wrap_at('180d'), Ms_co.b,s=50, color='fuchsia', marker='20')
            
            
            ax[1].scatter(Ms[0,4],Ms[0,5],s=200,color='fuchsia',zorder=3)
            MS_stars = SkyCoord(ra = Ms[0,4], dec = Ms[0,5],unit='degree',frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
            cluster_stars = SkyCoord(ra = datos[:,5][colores_index[i]], dec = datos[:,6][colores_index[i]],unit='degree',frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
            for estrella in range(len(cluster_stars)):
                sep_ban = MS_stars.separation(cluster_stars[estrella])
                if min(sep_ban.value) <1/3600:
                    ax[1].set_facecolor('lavender')
            # if len(index1[0]) > 0:
            ax[2].scatter((catal[:,3][index1]-catal[:,4][index1]),catal[:,4][index1], color='fuchsia',s=100,marker='2',zorder=3) 
           
            ax[2].scatter(datos[:,3][colores_index[i][0]]-datos[:,4][colores_index[i][0]],datos[:,4][colores_index[i][0]], color=color_de_cluster ,s=120,zorder=3, alpha=1)
            ax[2].invert_yaxis()  
            ax[2].set_xlabel('$H-Ks$',fontsize =30)
            ax[2].set_ylabel('$Ks$',fontsize =30)
            
            txt_color = '\n'.join(('H-Ks =%.3f'%(np.median(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
                                 '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
                                 'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))))
            props = dict(boxstyle='round', facecolor=color_de_cluster, alpha=0.3)
            # # place a text box in upper left in axes coords
            ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
                verticalalignment='top', bbox=props)
            
            ax[2].scatter(datos[:,3]-datos[:,4],datos[:,4],alpha=0.1,color ='k')
            ax[2].scatter(datos[:,3][group_md]-datos[:,4][group_md],datos[:,4][group_md],alpha=0.7,c='r',marker = 'x')
            txt_around = '\n'.join(('H-Ks =%.3f'%(np.median(datos[:,3][group_md]-datos[:,4][group_md])),
                                 '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][group_md]-datos[:,4][group_md])),
                                 'diff_color = %.3f'%(max(datos[:,3][group_md]-datos[:,4][group_md])-min(datos[:,3][group_md]-datos[:,4][group_md]))))
            props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
            ax[2].text(0.50, 0.25,txt_around, transform=ax[2].transAxes, fontsize=30,
                verticalalignment='top', bbox=props_arou)
       
        # %%
print(r_u[r]**2/3600)
      # 
       
        
       
        