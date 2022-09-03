#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:26:41 2022

@author: amartinez
"""

# =============================================================================
# Here we are going to look for clusters in simulated data, generated randomly with 
# no real clusters in it. Then, some how, I have to find a way to stract an statistic
# result out of it, that will allow me to sell my results
# =============================================================================
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from matplotlib.ticker import FormatStrFormatter

import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from sklearn.preprocessing import StandardScaler

from scipy.stats import gaussian_kde

import astropy.coordinates as ap_coor
import time

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
pruebas ='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
sim_dir ='/Users/amartinez/Desktop/PhD/Libralato_data/simulated_stat/'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
carpeta = '/Users/amartinez/Desktop/PhD/Libralato_data/regions_for_simulations/'
#Load a region generated in dbs_kernel_subsecA.py
# This is a chunck of real Libralato data

# dmu_lim = 2
# area = 2.1
# section ='A'
# sub_sec = '3_3' 
section = input('section =')
area = input('area =')
sub_sec = input('subsection =') 
dmu_lim = input('dmu_lim =')
simulated_by = input('Simulated by (kern or shuff):')
samples_dist = input('Samples_dist(dbscan parameter =')
samples_dist =int(samples_dist)
#    0         1      2        3       4    5    6      7    8     9      10    11   12      13    14   15   16    17    18    19     20     22      23        
#"'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
data = np.loadtxt(carpeta + 'sec%s_area%s_%s_dmu%s.txt'%(section,area,sub_sec,dmu_lim))

mul = data[:,17]
mub = data[:,18]

ra = data[:,0]
dec = data[:,1]
x,y = data[:,7], data[:,8]

H = data[:,3]
K = data[:,4]


#Coordinates in galactic
coordenadas = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')#
gal_c=coordenadas.galactic
t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))

fig, ax = plt.subplots(1,3, figsize =(30,10))
ax[0].scatter(mul,mub,alpha =0.05)
ax[0].set_xlim(-15,10)
ax[0].invert_xaxis()
# ax[1].scatter(t_gal['l'],t_gal['b'])
ax[1].scatter(x,y)
ax[2].scatter(H-K,K,alpha =0.1)
ax[2].set_xlim(1.2,3)
ax[2].invert_yaxis()

# %%
# Now we generate the simulated data. Of all the simulations we will select as
# the real data those with the minimun value of K-NN
# =============================================================================
# Note to myself: tried generated the simulated data with the kernel
# and with the suffle. Is something weird about the kernnel for the color...
# =============================================================================

# I going to make a loop a save the statistisc of the simuated clusters, and see
sim_clusted_stat =[]
long_bucle = 10000
tic = time.perf_counter()
for bucle in range(long_bucle):
    dic_Xsim = {} 
    dic_Knn = {}
    # samples_dist = 9
    clustered_by = 'all_color'#TODO we can choose look for clustes in 5D(all_color -> pm, position and color) or in 4D(all -> pm and position)
    # clustered_by = 'all'#TODO
    
    # simulated_by = 'kern'#TODO
    # simulated_by = 'shuff'#TODO
    lst_d_KNN_sim=[]#here we stored the minimun distance of the k-NN value for each simulation
    if simulated_by == 'kern':
        colorines = H - K
        mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
        x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
        ra_kernel, dec_kernel = gaussian_kde(ra), gaussian_kde(dec)
        color_kernel = gaussian_kde(colorines)
        for d in range(20):
            mub_sim,  mul_sim = mub_kernel.resample(len(data)), mul_kernel.resample(len(data))
            x_sim, y_sim = x_kernel.resample(len(data)), y_kernel.resample(len(data))
            ra_sim, dec_sim = ra_kernel.resample(len(data)), dec_kernel.resample(len(data))
            color_sim = color_kernel.resample(len(data))
            if clustered_by == 'all_color':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim[0],mub_sim[0],ra_sim[0],dec_sim[0],color_sim[0]]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                dic_Xsim['Xsim_%s'%(d)] = X_sim
                dic_Knn['Knn_%s'%(d)] = d_KNN_sim
            if clustered_by == 'all':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim[0],mub_sim[0],ra_sim[0],dec_sim[0]]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                dic_Xsim['Xsim_%s'%(d)] = X_sim
                dic_Knn['Knn_%s'%(d)] = d_KNN_sim
    if simulated_by == 'shuff':
        for d in range(20):
            randomize = np.arange(len(data))
            np.random.shuffle(randomize)
            mul_sim,  mub_sim = mul[randomize], mub[randomize]
            x_sim, y_sim = x, y
            ra_sim, dec_sim = ra, dec
            random_col = np.arange(len(data))
            np.random.shuffle(random_col)
            H_sim, K_sim = H[random_col], K[random_col]
            color_sim = H_sim-K_sim
            if clustered_by == 'all_color':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim,mub_sim,ra_sim,dec_sim,color_sim]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                dic_Xsim['Xsim_%s'%(d)] = X_sim
                dic_Knn['Knn_%s'%(d)] = d_KNN_sim
            if clustered_by == 'all':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim,mub_sim,ra_sim,dec_sim]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                dic_Xsim['Xsim_%s'%(d)] = X_sim
                dic_Knn['Knn_%s'%(d)] = d_KNN_sim
    
    d_KNN_min = min(lst_d_KNN_sim)
    d_KNN_max = max(lst_d_KNN_sim)
    # Retrieves the data set with the minumun K_NN that we will play the roll of real data
    real = np.argmin(lst_d_KNN_sim)
    simu = np.argmax(lst_d_KNN_sim)
    X = dic_Xsim['Xsim_%s'%(real)]
    d_KNN = dic_Knn['Knn_%s'%(real)]
    d_KNN_sim = dic_Knn['Knn_%s'%(simu)]
    eps_av = np.average([d_KNN_max,d_KNN_min])
    # Plotting the histogram for K-NN
# =============================================================================
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     # ax[0].set_title('Sub_sec_%s_%s'%(col[colum],row[ro]))
#     # ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
#     # ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')
#     
#     # # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
#     # ax[0].set_xlabel('Point') 
#     # ax[0].set_ylabel('%s-NN distance'%(samples)) 
#     
#     ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
#     ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
#     ax.set_xlabel('%s-NN distance'%(samples_dist)) 
#     
#     
#     texto = '\n'.join(('min d_KNN = %s'%(round(d_KNN_min,3)),
#                         'max d_KNN =%s'%(round(d_KNN_max,3)),'average = %.3f'%(eps_av)))
#     
#     
#     props = dict(boxstyle='round', facecolor='w', alpha=0.5)
#     # place a text box in upper left in axes coords
#     ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
#         verticalalignment='top', bbox=props)
#     
#     ax.set_ylabel('N') 
#     ax.set_xlim(0,2)
#     
#     plt.show()
# =============================================================================
    #Generates simulated coordinates for looking around later on
    coor_sim = SkyCoord(ra=X[:,2]*u.degree, dec=X[:,3]*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
    # =============================================================================
    # DBSCAN part
    # =============================================================================
    X_stad = StandardScaler().fit_transform(X)
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
        sim_clusted_stat.append(X[colores_index[i]])
        
# =============================================================================
#         fig, ax = plt.subplots(1,3,figsize=(30,10))
#         color_de_cluster = 'lime'
#         # fig, ax = plt.subplots(1,3,figsize=(30,10))
#         # ax[2].invert_yaxis()
#         
#         ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN_sim),3),clustered_by))
#         # t_gal['l'] = t_gal['l'].wrap_at('180d')
#         ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
#         ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
#         # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
#     
#         ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=color_de_cluster ,s=50,zorder=3)
#         ax[0].set_xlim(-10,10)
#         ax[0].set_ylim(-10,10)
#         ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
#         ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
#         ax[0].invert_xaxis()
#         ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
#         
#         mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
#         mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
#         
#         mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
#         mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
#     
#     
#         vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
#                              '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
#         vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
#                              '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
#         
#         propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
#         propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
#         ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
#             verticalalignment='top', bbox=propiedades)
#         ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
#             verticalalignment='top', bbox=propiedades_all)
#         
#        
#         
#         
#         #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
#         c2 = SkyCoord(ra = X[:,2][colores_index[i]]*u.deg,dec = X[:,3][colores_index[i]]*u.deg,frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
#         sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
#         rad = max(sep)/2
#         
#         radio_MS = max(sep)
#         
#         # This search for all the points around the cluster that are no cluster
#         lista = []
#         lista =np.zeros([len(c2),3])
#         # for c_memb in range(len(c2)):
#         #     distancia = list(c2[c_memb].separation(c2))
#         #     # print(int(c_memb),int(distancia.index(max(distancia))),max(distancia).value)
#         #     # a =int(c_memb)
#         #     # b = int(distancia.index(max(distancia)))
#         #     lista[c_memb][0:3]= int(c_memb),int(distancia.index(max(distancia))),max(distancia).value
#         
#         # coord_max_dist = list(lista[:,2]).index(max(lista[:,2]))
#     
#         # p1 = c2[int(lista[coord_max_dist][0])]
#         # p2 = c2[int(lista[coord_max_dist][1])]
#     
#         # m_point = SkyCoord(ra = [(p2.ra+p1.ra)/2], dec = [(p2.dec +p1.dec)/2])
#         
#         m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)],frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
#         
#         idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coor_sim, rad*2)
#         
#         ax[0].scatter(X[:,0][group_md],X[:,1][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)
#     
#         prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
#         ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
#                                 verticalalignment='top', bbox=prop)
#         
#         ax[1].scatter(X[:,2], X[:,3], color='k',s=50,zorder=1,alpha=0.01)#
#         ax[1].scatter(X[:,2][colores_index[i]],X[:,3][colores_index[i]],color=color_de_cluster ,s=50,zorder=3)
#         
#         
#         ax[1].scatter(X[:,2][group_md],X[:,3][group_md],s=50,color='r',alpha =0.1,marker ='x')
#         ax[1].set_xlabel('Ra(deg)',fontsize =30) 
#         ax[1].set_ylabel('Dec(deg)',fontsize =30) 
#         ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         # ax[1].set_title('col_row %.0f, %.0f.(%.2farcmin$^{2}$),Clus = %s'%(ic/0.5,jr/0.5,area,clus_num))
#        
#             
#         if simulated_by == 'shuff':
#             ax[2].scatter(H_sim - K_sim, K_sim, color = 'k', alpha = 0.05)
#             # ax[2].scatter(H - K, K, color = 'k', alpha = 0.05)
#             ax[2].invert_yaxis()
#             ax[2].set_xlim(1.2,2.5)
#         plt.show()
# =============================================================================
    if bucle%500 == 0:
        print(30*'+')
        print('Bucle #%s'%(bucle))
        print(30*'+')
toc = time.perf_counter()
print('Performing %s loops took %.0f seconds'%(long_bucle,toc-tic))
# %%
sigm_values, mean_values = np.zeros((len(sim_clusted_stat),5)), np.zeros((len(sim_clusted_stat),5))
for i in range(len(sim_clusted_stat)):
    clus_sta = np.vstack(sim_clusted_stat[i])
    sigm_values[i] = np.std(clus_sta[:,0]),np.std(clus_sta[:,1]),np.std(clus_sta[:,2]),np.std(clus_sta[:,3]),np.std(clus_sta[:,4])
    mean_values[i] = np.mean(clus_sta[:,0]),np.mean(clus_sta[:,1]),np.mean(clus_sta[:,2]),np.mean(clus_sta[:,3]),np.mean(clus_sta[:,4])
np.savetxt(sim_dir + 'sec%s_%s_std_area%s_dmul%s_%sims_%s.txt'%(section,sub_sec,
                                                             area,dmu_lim,long_bucle,simulated_by), np.array(mean_values),fmt = '%.5f', header ='mul, mub, ra, dec, color')  
np.savetxt(sim_dir + 'sig_sec%s_%s_std_area%s_dmul%s_%sims_%s.txt'%(section,sub_sec,
                                                             area,dmu_lim,long_bucle,simulated_by), np.array(sigm_values),fmt = '%.5f', header ='sig_mul, sig_mub, sig_ra, sig_dec, sig_color')
# %%
# 1
fig, ax = plt.subplots(1,2, figsize=(20,10))
gra = 0
eje = 0
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\mu}_{l}$',fontsize = 30)

gra = 1
eje = 1
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\mu}_{b}$',fontsize = 30)
# %
# 2
fig, ax = plt.subplots(1,2, figsize=(20,10))
gra = 0
eje = 2
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Ra',fontsize = 30)

gra = 1
eje = 3
ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Dec',fontsize = 30)
# %
# 3
fig, ax = plt.subplots(1,2, figsize=(20,10))

gra = 0
eje = 4
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(mean_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(mean_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]-np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(mean_values[:,eje])))
ax[gra].axvline(np.mean(mean_values[:,eje]+np.std(mean_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('Color (H - Ks)',fontsize = 30)

# %
# 4
fig, ax = plt.subplots(1,2, figsize=(20,10))


gra = 0
eje = 0
ax[gra].set_title('Sec%s_%s, area%s, dmul: %s, method: %s '%(section, sub_sec, area, dmu_lim,simulated_by))

ax[gra].hist(sigm_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(sigm_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]-np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]+np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\sigma}_{l}$',fontsize = 30)

gra = 1
eje = 1
ax[gra].hist(sigm_values[:,eje], bins ='auto')
ax[gra].axvline(np.mean(sigm_values[:,eje]), color ='k', linewidth =3, 
              label = 'mean %.2f'%(np.mean(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]-np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3,
              label = '$\sigma \pm$ %.2f'%(np.std(sigm_values[:,eje])))
ax[gra].axvline(np.mean(sigm_values[:,eje]+np.std(sigm_values[:,eje])),ls = '--', color ='k', linewidth =3)
ax[gra].legend()
ax[gra].set_xlabel('$\overline{\sigma}_{b}$',fontsize = 30)
      
      
      
      
      
      
