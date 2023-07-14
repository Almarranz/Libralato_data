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
import random
import alphashape
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
sim_dir ='/Users/amartinez/Desktop/PhD/Libralato_data/simulated_no_cluster/'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
carpeta = '/Users/amartinez/Desktop/PhD/Libralato_data/regions_for_simulations/'
#Load a region generated in dbs_kernel_subsecA.py
# This is a chunck of real Libralato data


# =============================================================================
# section = input('section =')
# area = input('area =')
# sub_sec = input('subsection =') 
# dmu_lim = input('dmu_lim =')
# simulated_by = input('Simulated by (kern or shuff):')
# samples_dist = input('Samples_dist(dbscan parameter =')
# samples_dist =int(samples_dist)
# =============================================================================

simulated_by = 'kern'
samples_dist = 15
clustered_by = 'all_color'#TODO we can choose look for clustes in 5D(all_color -> pm, position and color) or in 4D(all -> pm and position)
# clustered_by = 'all'#TODO
radio = 50*u.arcsec

#    0         1      2        3       4    5    6      7    8     9      10    11   12      13    14   15   16    17    18    19     20     22      23        
#"'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
# data = np.loadtxt(carpeta + 'sec%s_area%s_%s_dmu%s.txt'%(section,area,sub_sec,dmu_lim))
name='WFC3IR'
trimmed_data='no'
only_match = 'yes'
trimmed_data = 'yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
data_all=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
# data_all=np.loadtxt(results + 'sec_B_relaxed_match_GNS_and_WFC3IR_refined_galactic.txt')

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '

# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(data_all[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    data_all=data_all[valid]
    center=np.where(data_all[:,-2]-data_all[:,4]>3) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(data_all[:,3])==False) & (np.isnan(data_all[:,4])==False ))
    data_all=data_all[valid]
    center=np.where(data_all[:,3]-data_all[:,4]>1.3)
data_all=data_all[center]
dmu_lim = 1
vel_lim = np.where((data_all[:,19]<=dmu_lim) & (data_all[:,20]<=dmu_lim))
data_all=data_all[vel_lim]


# This choose a random point in the catalog and select stars around it
# =============================================================================
# rand_point = random.choice(np.arange(len(data_all)))
# coordenadas = SkyCoord(ra=data_all[:,0]*u.degree, dec=data_all[:,1]*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')#
# rand_coord = SkyCoord(ra = [coordenadas[rand_point].ra], dec = [coordenadas[rand_point].dec])
# idxc, group, d2d,d3d = coordenadas.search_around_sky(rand_coord,radio)
# =============================================================================

# This will select stars aroun a choosen massive stars
yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
m_star = [14996]
ind = np.where(yso[:,2] == m_star[0])[0][0]
print(yso[ind])
coordenadas = SkyCoord(ra=data_all[:,0]*u.degree, dec=data_all[:,1]*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')#
rand_coord = SkyCoord(ra = [yso[ind][0]], dec = [yso[ind][1]], unit = 'degree')
idxc, group, d2d,d3d = coordenadas.search_around_sky(rand_coord,radio)



data = data_all[group]
# sys.exit('121')
# mul = data[:,17]
# mub = data[:,18]

mul_or = data[:,9]
mub_or = data[:,11]
# 
ra = data[:,0]
dec = data[:,1]
x,y = data[:,7], data[:,8]

H = data[:,3]
K = data[:,4]
saving_esta = 'yes'#TODO
if saving_esta == 'no':
    answer = input(' You are not saving youir data\n Continue?')
elif saving_esta == 'yes':
    answer =  input(' You are about to save\n Continue?')

if answer == 'no':
    sys.exit('You stoped the thing')
if saving_esta == 'yes':
    with open(pruebas + 'clus_from_simul_ID%sknn%s_rad%s.txt'%(m_star[0],samples_dist,radio.value),'w') as file:
        file.write('# mu_ra 0, mu_dec 1, sig(mu_Ra) 2, sig(mu_Dec) 3, area 4, #stars 5, H-Ks 6, sigma (H-Ks) 7,delta(H-Ks) 8, densidad 9, loop 10, bucle 11 \n')
# I going to make a loop a save the statistisc of the simuated clusters, and see
sim_clusted_stat =[]
long_bucle = 500
tic = time.perf_counter()
for bucle in range(long_bucle):

    # We are going to use the shuffle to generate the rela population.
    # This will destroy any real cluster in the data.
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    mul, mub = mul_or[randomize], mub_or[randomize]
    # mul, mub = mul_or, mub_or
    color_real = H-K
    
    
    if clustered_by == 'all_color':
        # X=np.array([mul,mub,x,y,color_real]).T
        X=np.array([mul,mub,ra,dec,color_real]).T
        # X_stad = RobustScaler(quantile_range=(25, 75)).fit_transform(X)#TODO
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    elif clustered_by == 'all':
        # X=np.array([mul,mub,x,y]).T
        X=np.array([mul,mub,ra,dec]).T
        
        X_stad = StandardScaler().fit_transform(X)
        tree = KDTree(X_stad, leaf_size=2) 
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
    
    # sys.exit(126)
    #Coordinates in galactic
    coordenadas = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')#
    gal_c=coordenadas.galactic
    t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))
# %%
# =============================================================================
#     fig, ax = plt.subplots(1,3, figsize =(30,10))
#     ax[0].scatter(mul,mub,alpha =0.05)
#     ax[0].set_xlim(-15,10)
#     ax[0].invert_xaxis()
#     ax[1].scatter(data_all[:,0],data_all[:,1],color = 'k')
#     ax[1].scatter(ra,dec)
#     ax[2].scatter(H-K,K,alpha =0.1)
#     ax[2].set_xlim(1.2,3)
#     ax[2].invert_yaxis()
# =============================================================================
    # sys.exit(141)
    # %%
    # Now we generate the simulated data. Of all the simulations we will select as
    # the real data those with the minimun value of K-NN
    # =============================================================================
    # Note to myself: tried generated the simulated data with the kernel
    # and with the suffle. Is something weird about the kernnel for the color...
    # =============================================================================


    # dic_Xsim = {} 
    # dic_Knn = {}
    # samples_dist = 9
    
    
    # simulated_by = 'kern'#TODO
    # simulated_by = 'shuff'#TODO
    lst_d_KNN_sim=[]#here we stored the minimun distance of the k-NN value for each simulation
    if simulated_by == 'kern':
        colorines = H - K
        mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
        x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
        ra_kernel, dec_kernel = gaussian_kde(ra), gaussian_kde(dec)
        color_kernel = gaussian_kde(colorines)
        for d in range(10):
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
                # dic_Xsim['Xsim_%s'%(d)] = X_sim
                # dic_Knn['Knn_%s'%(d)] = d_KNN_sim
            if clustered_by == 'all':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim[0],mub_sim[0],ra_sim[0],dec_sim[0]]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                # dic_Xsim['Xsim_%s'%(d)] = X_sim
                # dic_Knn['Knn_%s'%(d)] = d_KNN_sim
    
    if simulated_by == 'shuff':
        for d in range(10):
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
                # dic_Xsim['Xsim_%s'%(d)] = X_sim
                # dic_Knn['Knn_%s'%(d)] = d_KNN_sim
            if clustered_by == 'all':
                # X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                X_sim=np.array([mul_sim,mub_sim,ra_sim,dec_sim]).T
    
                X_stad_sim = StandardScaler().fit_transform(X_sim)
                tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                
                dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim.append(min(d_KNN_sim))
                # dic_Xsim['Xsim_%s'%(d)] = X_sim
                # dic_Knn['Knn_%s'%(d)] = d_KNN_sim
    
    d_KNN_min = min(lst_d_KNN_sim)
    d_KNN_max = max(lst_d_KNN_sim)
    # Retrieves the data set with the minumun K_NN that we will play the roll of real data
    # real = np.argmin(lst_d_KNN_sim)
    # simu = np.argmax(lst_d_KNN_sim)
    # Selects a random value as the real data and the mean as the simulated data.
    real = np.argmin(d_KNN)
    # simu = random.choice(np.arange(len(lst_d_KNN_sim)))
    
    # sys.exit('244')
    # X = dic_Xsim['Xsim_%s'%(real)]
    # d_KNN = dic_Knn['Knn_%s'%(real)]
    # d_KNN_sim = dic_Knn['Knn_%s'%(simu)]
    eps_av_sim = np.mean([d_KNN_max,d_KNN_min])
    eps_av = np.mean([d_KNN[real],eps_av_sim])
    # Plotting the histogram for K-NN
# =============================================================================
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     
#     ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
#     ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
#     ax.set_xlabel('%s-NN distance'%(samples_dist)) 
#     
#     
#     texto = '\n'.join(('Real d_KNN = %.3f'%(d_KNN[real]),'min d_KNN = %s'%(round(d_KNN_min,3)),
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
    # sys.exit('277')
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
#         # ax[0].set_xlim(-10,10)
#         # ax[0].set_ylim(-10,10)
#         ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
#         # ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
#         ax[0].invert_xaxis()
#         # ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
# =============================================================================
        
        mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
        mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
        
        mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
        mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
    
    
# =============================================================================
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
# =============================================================================
        
        
        #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
        c2 = SkyCoord(ra = X[:,2][colores_index[i]]*u.deg,dec = X[:,3][colores_index[i]]*u.deg,frame ='icrs', equinox = 'J2000', obstime = 'J2014.2')
        sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
        rad = max(sep)/2
        
        radio_MS = max(sep)
        
        # This search for all the points around the cluster that are no cluster
        # lista = []
        # lista =np.zeros([len(c2),3])
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
        
# =============================================================================
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
# =============================================================================
        
        p2d = np.array([X[:,2][colores_index[i]],X[:,3][colores_index[i]]]).T
        
        al_sh = alphashape.alphashape(p2d,400)
        
        tipo = al_sh.type
        if tipo == 'MultiPolygon':
            contorno = {}
            for ci,cont in enumerate(al_sh.geoms):
                contorno['cont%s'%(ci)] = al_sh.geoms[ci].exterior.coords[:]
        elif tipo != 'MultiPolygon':
            contorno = np.array(al_sh.exterior.coords[:])
        supf = al_sh.area*3600
        
        
# =============================================================================
#         ax[1].scatter(X[:,2][group_md],X[:,3][group_md],s=50,color='r',alpha =0.1,marker ='x')
#         ax[1].set_xlabel('Ra(deg)',fontsize =30) 
#         ax[1].set_ylabel('Dec(deg)',fontsize =30) 
#         ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         # ax[1].set_title('col_row %.0f, %.0f.(%.2farcmin$^{2}$),Clus = %s'%(ic/0.5,jr/0.5,area,clus_num))
#         txt_around = '\n'.join(('$\overline{H-Ks}$ =%.2f'%(np.median(H[group_md]-K[group_md])),
#                              '$\sigma_{(H-Ks)}$ = %.2f'%(np.std(H[group_md]-K[group_md])),
#                              '$\Delta$ (H-Ks) = %.2f'%(max(H[group_md]-K[group_md])-min(H[group_md]-K[group_md]))))
#         props_arou = dict(boxstyle='round', facecolor='red', alpha=0.2)
#         ax[2].text(0.65, 0.19,txt_around, transform=ax[2].transAxes, fontsize=20,
#             verticalalignment='top', bbox=props_arou)
#             
#         txt_color = '\n'.join(('$\overline{H-Ks}$ =%.2f'%(np.median(H[colores_index[i]]-K[colores_index[i]])),
#                              '$\sigma_{(H-Ks)}$ = %.2f'%(np.std(H[colores_index[i]]-K[colores_index[i]])),
#                              '$\Delta(H-Ks)$ = %.2f'%(max(H[colores_index[i]]-K[colores_index[i]])-min(H[colores_index[i]]-K[colores_index[i]]))))
#         
#         props = dict(boxstyle='round', facecolor=color_de_cluster, alpha=0.2)
#         # # place a text box in upper left in axes coords
#         ax[2].text(0.50, 0.95, txt_color, transform=ax[2].transAxes, fontsize=30,
#             verticalalignment='top', bbox=props)
#         ax[2].scatter(H - K, K, color = 'k', alpha = 0.05)
#         ax[2].scatter(H[colores_index[i]] - K[colores_index[i]], K[colores_index[i]], color = color_de_cluster,s=50,zorder=3)
#         ax[2].scatter(H[group_md] - K[group_md], K[group_md], color = 'red', alpha = 0.05)
#         ax[2].invert_yaxis()
#         ax[2].set_xlim(1.5,3.5)
# =============================================================================
        densidad = len(colores_index[i][0])/supf
# =============================================================================
#         if tipo == 'MultiPolygon':
#             for cd in range(len(contorno)):
#                 ax[1].plot(np.array(contorno['cont%s'%(cd)])[:,0],np.array(contorno['cont%s'%(cd)])[:,1], label = 'Dens = %.0f stars/min2'%(densidad))
#             print()
#         elif tipo != 'MultiPolygon':
#             ax[1].plot(contorno[:,0],contorno[:,1], label = 'Dens = %.0f stars/min2'%(densidad))
#         ax[1].legend(loc = 3)
#         plt.show()
# =============================================================================
        if saving_esta == 'yes':
            with open(pruebas + 'clus_from_simul_ID%sknn%s_rad%s.txt'%(m_star[0],samples_dist,radio.value),'a') as file:
                file.write('%.4f %.4f %.4f %.4f %.1f %.0f %.4f %.4f %.4f %.1f %.0f %.0f\n'%( mul_mean, mub_mean,mul_sig, mub_sig,
                                                                                          supf,len(colores_index[i][0]),
                                                                                            np.median(H[colores_index[i]]-K[colores_index[i]]),np.std(H[colores_index[i]]-K[colores_index[i]]),max(H[colores_index[i]]-K[colores_index[i]])-min(H[colores_index[i]]-K[colores_index[i]]),
                                                                                            densidad,i, bucle))

    if bucle%50 == 0:
        print(30*'+')
        print('Bucle #%s'%(bucle))
        print(30*'+')
toc = time.perf_counter()
print('Performing %s loops took %.0f seconds'%(long_bucle,toc-tic))
sys.exit('512')
# %%
from matplotlib.colors import LinearSegmentedColormap

# mu_ra 0, mu_dec 1, sig(mu_Ra) 2, sig(mu_Dec) 3, area 4, #stars 5, H-Ks 6, sigma (H-Ks) 7,delta(H-Ks) 8, densidad 9, loop 10, bucle 11
sim_data = np.loadtxt(pruebas  + 'clus_from_simul_ID%sknn%s_rad%s.txt'%(m_star[0],samples_dist,radio.value))
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(89, 317, marker = 'x', color = 'red')
im = ax.hist2d(sim_data[:,5],sim_data[:,9],bins = (10,20), cmap = 'winter')
ax.set_xlabel('# stars')
ax.set_ylabel('Density (# stars/min$^{2}$)')
fig.colorbar(im[3], ax=ax)
ax.set_xlim(0,100)
ax.set_ylim(100,400)
ax.axvline(samples_dist, color = 'r',linestyle ='dashed')
# ax.set_aspect('equal',adjustable ='box')
# plt.colorbar(im, ax =ax)

# %%
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(sim_data[:,5],sim_data[:,9], cmap = 'Greys')
ax.scatter(89, 317, marker = 'x', color = 'red')
ax.set_xlabel('# stars')
ax.set_ylabel('$\\rho$ (# stars/min$^{2}$)')
ax.axvline(samples_dist, color = 'r')
# fig.colorbar(im[3], ax=ax)
# ax.set_ylim(0,400)
# %%
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis,markersize=20)
    fig.colorbar(density, label='Number of points per pixel')

fig = plt.figure()
using_mpl_scatter_density(fig,sim_data[:,5], sim_data[:,9])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate fake data
x = sim_data[:,5]
y = sim_data[:,9]

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=z, s=50)
ax.scatter(89, 317, marker = 'x', color = 'red')
fig.colorbar(im)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('$\\rho$')

    return ax




density_scatter( x , y, ax = None, sort = True, bins = 20 ) 


