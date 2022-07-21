#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:01:02 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.coordinates as ap_coor
import pandas as pd
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
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
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'



# %%
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
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
section = 'A'#selecting the whole thing

if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%
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
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

color_de_cluster = 'lime'

# for clus_f in glob.glob(folder +'*'):
# % coordinates
ra_=catal[:,5]
dec_=catal[:,6]
# Process needed for the trasnformation to galactic coordinates
coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
gal_c=coordenadas.galactic

t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))

mul,mub = catal[:,-6],catal[:,-5]
x,y = catal[:,7], catal[:,8]
colorines = catal[:,3]-catal[:,4]
# %%Thi is for the extinction

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
# %%

# section_folder = '/Users/amartinez/Desktop/morralla/Sec_A_at_2022-07-19_all_and_allcolor/'
section_folder = '/Users/amartinez/Desktop/morralla/Sec_A_at_2022-07-20 12/'#Test folder


plots =0
for folder in sorted(glob.glob(section_folder + 'cluster_num*'),key = os.path.getmtime):
   print(folder)
   for cluster_f in sorted(glob.glob(folder + '/cluster*.txt'),key = os.path.getmtime):
       print(cluster_f)
   plots +=1
   print(30*'+')
print(plots)
# %%
plots =0

# for folder in sorted(glob.glob(section_folder + 'cluster_num*'),key = os.path.getmtime):
for folder in sorted(glob.glob(section_folder + 'cluster_num13_0_knn7_area2.12'),key = os.path.getmtime):#cluster for testing
 
    all_clus = []
    cluster_len =[]
    equal_method=[]
    clus_per_folder =0
    for file in glob.glob(folder + '/cluster*'):
        if 'all_color.txt' in os.path.basename(file):
            equal_method.append(0)
        elif 'all.txt' in os.path.basename(file):
            equal_method.append(1)
        clus_per_folder += 1  
        # ra, dec, l, b, pml, pmb,J, H, Ks,x, y, AKs_mean, dAks_mean, radio("),cluster_ID
        cluster = np.loadtxt(file)
        cluster_len.append(len(cluster))
        for line in range(len(cluster)):
            all_clus.append(cluster[line])
    clus_arr = np.array(all_clus)
    print(30*'∂')
    print(clus_per_folder)
    print(30*'∂')
# % 
    same_method = 'yes'
    if len(set(equal_method)) ==2:
        same_method = 'no'
        
    cu_test, index_u = np.unique(clus_arr[:,0:2],return_index=True,axis=0)
    cluster_unique = clus_arr[index_u]
    new_stars = 'no'
    if any(x<len(cluster_unique) for x in cluster_len):
        new_stars = 'yes'
        print('some stars added')
    else:
        print('all cluster are the same')
    plots += 1
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    ax[0].set_title('Plot #%s. Found %s times. Combiend cluster: %s'%(plots,clus_per_folder,new_stars))
    ax[0].scatter(catal[:,-6],catal[:,-5],color = 'k', alpha = 0.1, zorder=1)
    ax[0].scatter(cluster_unique[:,4],cluster_unique[:,5], color = color_de_cluster, zorder=3,s=100) 
    ax[0].set_xlim(-10,10)
    ax[0].set_ylim(-10,10)
    ax[0].invert_xaxis()
    
    mul_sig, mub_sig = np.std(cluster_unique[:,4]), np.std(cluster_unique[:,5])
    mul_mean, mub_mean = np.mean(cluster_unique[:,4]), np.mean(cluster_unique[:,5])
    
    mul_sig_all, mub_sig_all = np.std(catal[:,-6]), np.std(catal[:,-5])
    mul_mean_all, mub_mean_all = np.mean(catal[:,-6]), np.mean(catal[:,-5])
    
    
    vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
    vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
    
    propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
    propiedades_all = dict(boxstyle='round', facecolor='k', alpha=0.1)
    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
        verticalalignment='top', bbox=propiedades)
    ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
        verticalalignment='top', bbox=propiedades_all)
    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
    
    c2 = SkyCoord(ra = cluster_unique[:,0]*u.deg,dec = cluster_unique[:,1]*u.deg)
    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
    rad = max(sep)/2
    
    m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)])
    
    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*1.5)
    
    ax[0].scatter(catal[:,-6][group_md],catal[:,-5][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)

    prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
    ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(cluster_unique)), transform=ax[1].transAxes, fontsize=30,
                            verticalalignment='top', bbox=prop)
    
    # ax[1].scatter(catal[:,7][group_md],catal[:,8][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.3)

    # ax[1].scatter(catal[:,7],catal[:,8],color ='k',alpha = 0.1)
    # ax[1].scatter(cluster_unique[:,9], cluster_unique[:,10], color = color_de_cluster,s=100)
    # ax[1].set_xlabel('x',fontsize =30) 
    # ax[1].set_ylabel('y',fontsize =30) 
    ax[1].scatter(catal[:,5][group_md],catal[:,6][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.3)

    ax[1].scatter(ra_,dec_,color ='k',alpha = 0.1)
    ax[1].scatter(cluster_unique[:,0], cluster_unique[:,1], color = color_de_cluster,s=100)
    ax[1].set_xlabel('Ra',fontsize =30) 
    ax[1].set_ylabel('Dec',fontsize =30,labelpad = -10) 
    
    # Here we are going to match the cluster with gns extion for get the average extiontion and plot an isochrone
    H_Ks_yes = []
    Ks_yes = []
    AKs_clus_all =[]
    clus_coord = SkyCoord(ra = cluster_unique[:,0], dec = cluster_unique[:,1], unit = 'deg')
    idx = clus_coord.match_to_catalog_sky(gns_coord)
    validas = np.where(idx[1]<0.5*u.arcsec)
    gns_match = AKs_center[idx[0][validas]]
    for member in range(len(gns_match)):
        if gns_match[member,16] != '-' and gns_match[member,18] != '-':
            AKs_clus_all.append(float(gns_match[member,18]))
            
    AKs_clus, std_AKs = np.median(AKs_clus_all),np.std(AKs_clus_all)
    absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
    AKs = min(AKs_list, key=absolute_difference_function)
    
    iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
    dist = 8200 # distance in parsec
    metallicity = 0.30 # Metallicity in [M/H]
    # # logAge_600 = np.log10(0.61*10**9.)
    logAge = np.log10(0.010*10**9.)
    # logAge_30 = np.log10(0.030*10**9.)
    # logAge_60 = np.log10(0.060*10**9.)
    # logAge_90 = np.log10(0.090*10**9.)
    evo_model = evolution.MISTv1() 
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawNoguerasLara18()
    filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
    
    iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
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
    
    ####################################################
    ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
                      iso.points['m_hawki_Ks'], 'b-',  label='10 Myr')
    
    txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
    ax[2].text(0.65, 0.50, txt_AKs, transform=ax[2].transAxes, fontsize=20,
        verticalalignment='top', bbox=propiedades_all)
    
    ax[2].scatter(catal[:,3]-catal[:,4],catal[:,4], color='k' ,s=50,zorder=1, alpha=0.03)
    ax[2].scatter(catal[:,3][group_md]-catal[:,4][group_md],catal[:,4][group_md], color='r' ,s=50,zorder=1, alpha=0.5,marker='x')

    ax[2].scatter(cluster_unique[:,7]-cluster_unique[:,8],cluster_unique[:,8], color=color_de_cluster ,s=120,zorder=3, alpha=1)
    ax[2].invert_yaxis()  
    
    ax[2].set_xlabel('$H-Ks$',fontsize =30)
    ax[2].set_ylabel('$Ks$',fontsize =30)
    ax[2].set_xlim(1.3,2.5)
    ax[2].set_ylim(max(catal[:,4]),min(catal[:,4]))
    # ax[2].set_title('Cluster %s, eps = %s'%(clus_num,round(eps_av,3)))
    txt_clus = '\n'.join(('H-Ks =%.3f'%(np.mean(cluster_unique[:,7]-cluster_unique[:,8])),
                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(cluster_unique[:,7]-cluster_unique[:,8])),
                         'diff_color = %.3f'%(max(cluster_unique[:,7]-cluster_unique[:,8])-min(cluster_unique[:,7]-cluster_unique[:,8]))))
    props_arou = dict(boxstyle='round', facecolor=color_de_cluster, alpha=0.3)
    ax[2].text(0.45, 0.90,txt_clus, transform=ax[2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props_arou)
    
    txt_around= '\n'.join(('H-Ks =%.3f'%(np.mean(catal[:,3][group_md]-catal[:,4][group_md])),
                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(catal[:,3][group_md]-catal[:,4][group_md])),
                         'diff_color = %.3f'%(max(catal[:,3][group_md]-catal[:,4][group_md])-min(catal[:,3][group_md]-catal[:,4][group_md]))))
    props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
    ax[2].text(0.45, 0.30,txt_around, transform=ax[2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props_arou)
    # sys.exit('line 232')

# %%
cu_test, index = np.unique(clus_arr[:,0:2],return_index=True,axis=0)
print(clus_arr[index])
print(len(cu_test))

               




















