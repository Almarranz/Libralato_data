# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:00:21 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
from matplotlib import rcParams

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
# ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks, dAks
path_clus ='/Users/amartinez/Desktop/morralla/Sec_A_dmu1_at_2022-08-01/cluster_num0_0_knn10_area19.08/cluster0_0_0_knn_10_area_19.08_all_color.txt'
# path_clus = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/Cluster_virial_test/cluster_num1_0_knn7_area19.08/cluster1_0_0_knn_7_area_19.08.txt'
# ra, dec, l, b, pml, pmb,J, H, Ks,x, y

mul,mub, J,H,K,AKs_clus, std_AKs, radio_clus,ID =np.loadtxt(path_clus, usecols=(4,5,6,7,8,11,12,13,14),unpack=True)

sig_mul, sig_mub = np.std(mul), np.std(mub)

sig_mu = np.mean([sig_mul, sig_mub])
# sig_mu = sig_mul

AKs_list =  np.arange(1.6,2.11,0.01)

# %%
# <v2> = 0.4*GM/rh. 
# Where <v2> is the mean square velocity of the star system
# rh is the distance where lies half of the mass
# G ~ 0.0045 pc**3 M_sol**-1 Myr**-2


theta = radio_clus[0]*u.arcsec.to(u.rad)
dist = 8200*u.parsec
r_eff = theta*dist
rh = r_eff

# Here Im using the velocities dispersion instear of the velocities
# follow the aproach in http://spiff.rit.edu/classes/phys440/lectures/glob_clus/glob_clus.html
# But I´m far for be sure about it. The reallity is than if I use the real velocities the
# cluster mas is tooooooo high

sig_mu2 = (3*((sig_mu))**2*40)*(u.km/u.second)#0.0625 is std**2 (0.25**2)

mu_pc_myr = sig_mu2.to(u.pc/u.Myr)
G = 0.0045*(u.pc**3)*(u.Myr**-2)*(u.M_sun**-1)

M_clus = 0.4*(rh * mu_pc_myr**2)/G
print(M_clus)

# Now we are define the crossing time according with Mark Gieles et al. 2011
# We will discard the crossing time right now

# Tcr ≡  10*(r_eff**3/(GM))**0.5
# Tcr = 10*np.sqrt((r_eff**3)/(G*M_clus))
# print(Tcr)
# PI_2 = age2/Tcr
# print(PI_2)
# %%
# Try using spisea to calculate the min mass of the cluster 
# containing those stars

 


iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'

dist = 8000 # distance in parsec
metallicity = 0.30 # Metallicity in [M/H]
# logAge_600 = np.log10(0.61*10**9.)
logAge = np.log10(0.010*10**9.)
# logAge = np.log10(0.010*10**9.)
# logAge_30 = np.log10(0.030*10**9.)
# logAge_60 = np.log10(0.060*10**9.)
# logAge_90 = np.log10(0.090*10**9.)
evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()
filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']

absolute_difference_function = lambda list_value : abs(list_value - AKs_clus[0])

AKs = min(AKs_list, key=absolute_difference_function)

iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

# iso_30 = synthetic.IsochronePhot(logAge_30, AKs, dist, metallicity=metallicity,
#                                 evo_model=evo_model, atm_func=atm_func,
#                                 red_law=red_law, filters=filt_list,
#                                     iso_dir=iso_dir)
# iso_60 = synthetic.IsochronePhot(logAge_60, AKs, dist, metallicity=metallicity,
#                                 evo_model=evo_model, atm_func=atm_func,
#                                 red_law=red_law, filters=filt_list,
#                                     iso_dir=iso_dir)

# iso_90 = synthetic.IsochronePhot(logAge_90, AKs, dist, metallicity=metallicity,
#                                 evo_model=evo_model, atm_func=atm_func,
#                                 red_law=red_law, filters=filt_list,
#                                     iso_dir=iso_dir)
# # #%
# #%


imf_multi = multiplicity.MultiplicityUnresolved()

# # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01

# # NOTE: when defining the power law slope for each segment of the IMF, we define
# # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
# # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 

massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
# my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)


# #%


max_stars = len(K)**2
porcentaje = 0
while  max_stars > len(K)+0.3*len(K):

    # mass = 0.8*10**4.
    mass = M_clus.value -0.05*porcentaje*M_clus.value
    # dAks = std_AKs[0]
    dAks = 0.05

    cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
    cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
    clus = cluster.star_systems
    clus_ndiff = cluster_ndiff.star_systems
    
    max_mass = np.where((clus_ndiff['m_hawki_Ks']>min(K))&(clus_ndiff['m_hawki_Ks']<max(K)))
    
    max_stars = len(clus_ndiff['m_hawki_Ks'][max_mass])
    porcentaje +=1

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].hist(clus['mass'],bins = 'auto',color ='k')#, label ='Cluster Mass = %.0f$M_{\odot}$ \n virial mass = %.0f'%(mass,M_clus.value) )
ax[0].set_xlabel('$(M_{\odot})$')
ax[0].set_ylabel('$N$')
ax[0].set_title('Cluster %.0f, Radio = %.2f"'%(ID[0],radio_clus[0]))

# ax[1].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'slategray',alpha=0.7)
ax[1].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.6,s=50)
ax[1].invert_yaxis()
ax[1].scatter(H-K,K,color ='lime',s=100)
props = dict(boxstyle='round', facecolor='w', alpha=0.5)

ax[1].text(0.55, 0.95, 'L mass = %.0f $M_{\odot}$ \nV mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax[1].transAxes, fontsize=25,
    verticalalignment='top', bbox=props)
ax[1].set_xlabel('H-Ks')
ax[1].set_ylabel('Ks')
ax[1].set_title('[$\sigma_{mul}$= %.3f, $\sigma_{mub}$= %.3f]'%(sig_mul,sig_mub))
# txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
#                      'age = %.0f Myr'%(10**logAge/10**6)))
# txt_AKs = '\n'.join(('H-Ks =%.3f'%(np.mean(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))
#                      ,'AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))

    
plt.savefig('/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/' + 'cluster_virial_mass.png', dpi=300,bbox_inches='tight')    
# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=0.5)
ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.6,s=50)
ax[2].invert_yaxis()
ax[2].scatter(H-K,K,color ='lime',s=100)
props = dict(boxstyle='round', facecolor='w', alpha=0.5)

ax[2].text(0.55, 0.95, 'L mass = %.0f $M_{\odot}$ \nV mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax[2].transAxes, fontsize=25,
    verticalalignment='top', bbox=props)
ax[2].set_xlabel('H-Ks')
ax[2].set_ylabel('Ks')
ax[2].set_title('Cluster Radio = %.2f"'%(radio_clus[0]))
# txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
#                      'age = %.0f Myr'%(10**logAge/10**6)))
# txt_AKs = '\n'.join(('H-Ks =%.3f'%(np.mean(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      '$\sigma_{H-Ks}$ = %.3f'%(np.std(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])),
#                      'diff_color = %.3f'%(max(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]])-min(datos[:,3][colores_index[i]]-datos[:,4][colores_index[i]]))
#                      ,'AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# # place a text box in upper left in axes coords
# ax[2].text(0.65, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
#     verticalalignment='top', bbox=props)

# %
print(clus.columns)


ax[0].hist(clus['mass'],bins = 'auto',color ='k')#, label ='Cluster Mass = %.0f$M_{\odot}$ \n virial mass = %.0f'%(mass,M_clus.value) )
ax[0].set_xlabel('$(M_{\odot})$')
ax[0].set_ylabel('$N$')
ax[0].text(0.35, 0.65, 'L mass = %.0f $M_{\odot}$ \nV mass = %.0f $M_{\odot}$'%(mass, M_clus.value), transform=ax[0].transAxes, fontsize=25,
    verticalalignment='top', bbox=props)
ax[0].legend()
ax[0].set_xlim(0,2.5)
ax[0].set_title('$\sigma_{mul}$= %.3f, $\sigma_{mub}$= %.3f'%(sig_mul,sig_mub))
# fig, ax = plt.subplots(1,1,figsize=(10,10))
ax[1].scatter(clus_ndiff['mass'],clus_ndiff['m_hawki_Ks'],color ='k')
ax[1].set_xlabel('$(M_{\odot})$')
ax[1].set_ylabel('$Ks$')
ax[1].scatter(np.full(len(K),max(clus_ndiff['mass'])),K,color ='red')
ax[1].set_title('Cluster %.0f'%(ID[0]))
ax[1].invert_yaxis()

# ax.set_xlim(0,2.5)



# %%















