#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:44:43 2023

@author: amartinez
"""

# Dedicated script for calculated de Arches and Quintuplet masses based using 
# Spicea and extiontion maps (or lists for the stars with their individuals AJKs wheter works best)
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
from matplotlib import rcParams
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from extinction_simulation import extinction_now
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
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
# 
# choosen_cluster = 'Quintuplet'
choosen_cluster = 'Arches'

if choosen_cluster == 'Arches':
    path_clus ='/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/Arches_0.00evol_times_knn_30.txt'
    ra, dec, mura,mudec,f127, f153, H,K = np.loadtxt(path_clus,unpack=True)
elif choosen_cluster == 'Quintuplet':
    path_clus ='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/Quintuplet_0evoltime.txt'
    ra, dec, mura,mudec, f127, f153, H,K= np.loadtxt(path_clus,unpack=True)
else:
    path_clus ='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/ms_clus_14996.txt'
    ra, dec, mura,mudec,H,K = np.loadtxt(path_clus,unpack=True)



center = np.where(H-K>1.3)
ra	= ra[center]
dec	= dec[center]
mura 	= mura[center]
mudec =mudec[center]
f127	=f127[center]
f153	=f153[center]
H	=H[center]
K = K[center]

clean = np.where((np.isnan(K)==False) & (np.isnan(K)==False))
no_clean = np.where((np.isnan(K)==True) | (np.isnan(K)==True))
ra	=ra[clean]
dec	=dec[clean]
mura	=mura[clean]
mudec	=mudec[clean]
f127	=f127[clean]
f153	=f153[clean]
H	=H[clean]
K = K[clean]


AKs_list =  np.arange(1.6,2.63,0.01)
sig_mura,sig_mudec = np.std(mura), np.std(mudec)
# %%

# Extinction with maps
maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
layer =1
cor = np.array([ra,dec]).T

AH_clus_all, AKs_clus_all = extinction_now(cor,layer)

AKs_clus, dAks = np.mean(AKs_clus_all), np.std(AKs_clus_all)
# With coeficients
# =============================================================================
# gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
# Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)
# AKs_np = Aks_gns.to_numpy()#TODO
# center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
# AKs_center =AKs_np[center]#TODO
# # %
# AKs_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree,frame ='icrs', equinox = 'J2000', obstime = 'J2015.5')
# 
# clus_coord =  SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
# idx = clus_coord.match_to_catalog_sky(AKs_coord)
# validas = np.where(idx[1]<0.5*u.arcsec)
# gns_match = AKs_center[idx[0][validas]]
# AKs_clus_all =[] 
# for member in range(len(gns_match)):
#     if gns_match[member,18] != '-':
#         AKs_clus_all.append(float(gns_match[member,18]))
#         
# AKs = np.nanmean(AKs_clus_all)
# print(AKs)
# sys.exit('96')
# =============================================================================

iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'

dist = 8200 # distance in parsec
metallicity = 0.30 # Metallicity in [M/H]
# logAge_600 = np.log10(0.61*10**9.)
if choosen_cluster == 'Arches':
    logAge = np.log10(0.0025*10**9.)
elif choosen_cluster == 'Quintuplet':
    logAge = np.log10(0.005*10**9.)
# logAge = np.log10(0.010*10**9.)
# logAge_30 = np.log10(0.030*10**9.)
# logAge_60 = np.log10(0.060*10**9.)
# logAge_90 = np.log10(0.090*10**9.)
evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()
filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']

absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
AKs = min(AKs_list, key=absolute_difference_function)
# ind_ext = np.where(abs(AKs_list -AKs_clus) == min(abs(AKs_list - AKs_clus)))
# AKs =float(AKs_list[ind_ext])


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
my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
# my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)

mag_test = np.where((K>10)&(K<18))
print(max(K))
K = K[mag_test]
H = H[mag_test]
# #%

M_clus = 1*1e4*u.Msun
max_stars = len(K)**2
porcentaje = 0
while  max_stars > len(K)+0.1*len(K):

    # M_clus = 2*10**4*u.Msun
    mass = M_clus.value -0.01*porcentaje*M_clus.value
    dAks = np.std(AKs_clus_all)
    # dAks = 0.05

    cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
    cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
    clus = cluster.star_systems
    clus_ndiff = cluster_ndiff.star_systems
    
    # max_mass = np.where((clus_ndiff['m_hawki_Ks']>min(K))&(clus_ndiff['m_hawki_Ks']<max(K)))
    max_mass = np.where((clus['m_hawki_Ks']>min(K))&(clus['m_hawki_Ks']<max(K)))
    
    # max_stars = len(clus_ndiff['m_hawki_Ks'][max_mass])
    max_stars = len(clus['m_hawki_Ks'][max_mass])
    porcentaje +=1
# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))

ax.scatter(H-K,K,color ='#ff7f0e',s=100, label = '%s'%(choosen_cluster),zorder =3)

# ax.scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'slategray',alpha=0.7)
ax.scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],
           color = 'gray',alpha=1,s=50,zorder =3,linestyle="-", label = '$M_{model}$ = %.0f $M_{\odot}$'%(mass))
# ax.plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], iso.points['m_hawki_Ks'], 'k-')
# ax.scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=0.5,zorder =3)

all_color = clus['m_hawki_H']-clus['m_hawki_Ks']
min_col = []
max_col = []
ax.invert_yaxis()
# Ks_sor = np.arange(min(clus['m_hawki_Ks']),max(clus['m_hawki_Ks']),0.5)
Ks_sor = np.arange(min(np.round(np.round(clus['m_hawki_Ks'],0))),max(np.round(np.round(clus['m_hawki_Ks'],0))),0.5)
# Ks_sor = np.arange(10,19,0.5)

for j, inte in enumerate(Ks_sor):
    if j == 0:
        print('this is the lastone')
        mm = np.where((clus['m_hawki_Ks']>Ks_sor[j]) & (clus['m_hawki_Ks']<Ks_sor[j+1]))
        left = min(all_color[mm])
        right = max(all_color[mm])
        print(left, right) 
        print('whaaat')
        min_col.append(left)
        max_col.append(right)
        
    if j >0 and j<len(Ks_sor)-1:
        print(j)
        mm = np.where((clus['m_hawki_Ks']>Ks_sor[j-1]) & (clus['m_hawki_Ks']<Ks_sor[j+1]))
        left = min(all_color[mm])
        right = max(all_color[mm])
        min_col.append(left)
        max_col.append(right)
       
    
        
# ax.plot(min_col,Ks_sor[0:-1])
# ax.plot(max_col,Ks_sor[0:-1])
for j in range(len(Ks_sor)-1):
    # ax.axvspan(min_col[j], max_col[j],ymin=1-(1/(len(Ks_sor)-1))*(j+1), ymax = 1-(1/(len(Ks_sor)-1))*(j), 
    #            alpha=0.1, color='red', ls ='')
    ax.axvspan(min_col[j], max_col[j],ymin=1-0.05*(j+1), ymax = 1-0.05*(j), 
                alpha=0.3, color='gray', ls ='')
    if j == 0:
        ax.axvspan(min_col[j], max_col[j],ymin=1-0.05*(j+1), ymax = 1-0.05*(j), 
                alpha=0.3, color='gray', ls ='', label ='$\overline{AKs}$ = %.2f$\pm$%.2f'%(AKs,dAks))


props = dict(boxstyle='round', facecolor='w', alpha=0.5)
ax.legend()
# ax.text(0.55, 0.95, 'L mass = %.0f $M_{\odot}$ \n ini mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax.transAxes, fontsize=25,
#     verticalalignment='top', bbox=props)
ax.set_xlabel('H-Ks')
ax.set_ylabel('Ks')
ax.set_title('[$\sigma_{mul}$= %.3f, $\sigma_{mub}$= %.3f]'%(sig_mura,sig_mudec))
# ax.set_ylim(max(K),min(clus['m_hawki_Ks']))
ax.set_ylim(max(K),min(K))
# txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
#                      'age = %.0f Myr'%(10**logAge/10**6)))
# txt_AKs = '\n'.join(('H-Ks =%.3f'%(np.mean(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))
#                      ,'AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))

    
# plt.savefig('/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/' + 'cluster_virial_mass.png', dpi=300,bbox_inches='tight')    
# %
# fig, ax = plt.subplots(1,3,figsize=(30,10))

# ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=0.5)
# ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.6,s=50,zorder = 3)
# ax[2].invert_yaxis()
# ax[2].scatter(H-K,K,color ='lime',s=100)
# props = dict(boxstyle='round', facecolor='w', alpha=0.5)

# ax[2].text(0.55, 0.95, 'L mass = %.0f $M_{\odot}$ \nV mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax[2].transAxes, fontsize=25,
#     verticalalignment='top', bbox=props)
# ax[2].set_xlabel('H-Ks')
# ax[2].set_ylabel('Ks')
# # ax[2].set_title('Cluster Radio = %.2f"'%(radio_clus[0]))
# # txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
# #                      'age = %.0f Myr'%(10**logAge/10**6)))
# # txt_AKs = '\n'.join(('H-Ks =%.3f'%(np.mean(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
# #                      '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
# #                      'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))
# #                      ,'AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
# props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# # # place a text box in upper left in axes coords
# # ax[2].text(0.65, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
# #     verticalalignment='top', bbox=props)

# # %
# print(clus.columns)


# ax[0].hist(clus['mass'],bins = 'auto',color ='k')#, label ='Cluster Mass = %.0f$M_{\odot}$ \n virial mass = %.0f'%(mass,M_clus.value) )
# ax[0].set_xlabel('$(M_{\odot})$')
# ax[0].set_ylabel('$N$')
# ax[0].text(0.35, 0.65, 'L mass = %.0f $M_{\odot}$ \nV mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax[0].transAxes, fontsize=25,
#     verticalalignment='top', bbox=props)
# ax[0].legend()
# ax[0].set_xlim(0,2.5)
# ax[0].set_title('$\sigma_{mul}$= %.3f, $\sigma_{mub}$= %.3f'%(sig_mura,sig_mudec))
# # fig, ax = plt.subplots(1,1,figsize=(10,10))
# ax[1].scatter(clus_ndiff['mass'],clus_ndiff['m_hawki_Ks'],color ='k')
# ax[1].set_xlabel('$(M_{\odot})$')
# ax[1].set_ylabel('$Ks$')
# ax[1].scatter(np.full(len(K),max(clus_ndiff['mass'])),K,color ='red')
# # ax[1].set_title('Cluster %.0f'%(ID[0]))
# ax[1].invert_yaxis()

# # ax.set_xlim(0,2.5)









                            