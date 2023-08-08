#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:43:06 2023

@author: amartinez
"""
# We are going to calculate the qualities of the Hosek cluster based on the proba
# bilities values givem and compare then with qualities of one of the dbscan clus

# Here we are gooing to look for the Arches cluster using dbscan
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from astropy.io import ascii
import astropy.coordinates as ap_coor
import species
from matplotlib.ticker import FormatStrFormatter
import astroalign as aa
# %%plotting pa    metres
from matplotlib import rc
from matplotlib import rcParams
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
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# 

catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
morralla ='/Users/amartinez/Desktop/morralla/'
cls_to_throw = '/Users/amartinez/Desktop/PhD/Libralato_data/cluster_to_throw/'
pruebas = '/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

# 
choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO 
ref_frame = 'ecuatorial'#TODO
# ref_frame = 'galactic'#TODO

if choosen_cluster == 'Arches':
    arches = ascii.read(catal + 'Arches_from_Article.txt') 
if choosen_cluster == 'Quintuplet':
    arches = ascii.read(catal + 'Quintuplet_from_Article.txt') 

bgrd = np.where(arches['F127M']-arches['F153M']>1.7)
arches = arches[bgrd]
pme_lim = 1.4
valid_epm = np.where((arches['e_pmRA'] < pme_lim)&(arches['e_pmDE'] < pme_lim))
arches=arches[valid_epm]
center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet

RA_DEC = center_arc.spherical_offsets_by(arches['dRA'], arches['dDE'])
RA = RA_DEC.ra
DEC = RA_DEC.dec

# %%
alph = 0.01
bg_cl = 'k'
good_pro = np.where(arches['Pclust'] >=0.7)
fig, ax = plt.subplots(1,3,figsize=(30,10))
pmra, pmdec = arches['pmRA'][good_pro],arches['pmDE'][good_pro]
racl, decl = RA[good_pro], DEC[good_pro]
f127, f153 = arches['F127M'][good_pro], arches['F153M'][good_pro]
color = f127-f153
# Half-light radio
species.SpeciesInit()   
synphot = species.SyntheticPhotometry('HST/WFC3_IR.F153M')
# synphot = species.SyntheticPhotometry('Paranal/HAWKI.Ks')

RA_DEC_clus = center_arc.spherical_offsets_by(arches['dRA'][good_pro], arches['dDE'][good_pro])
cent_sep = center_arc.separation(RA_DEC_clus).arcsec
flux = np.array([synphot.magnitude_to_flux(f153[mag], error=0.0, zp_flux=None)[0] for mag in range(len(f153))])
clus_sep =np.c_[racl.value,decl.value,f153.value,flux,cent_sep]
# clus_sep =np.c_[racl.value,decl.value,f153,flux,cent_sep]
light = sum(flux)

clus_sep = clus_sep[clus_sep[:, -1].argsort()]
cum = np.cumsum(clus_sep[:,3])
norm_cum = cum/light
hl_ind = np.where(cum < light/2)
eff_rad = clus_sep[hl_ind[0][-1]][-1]
phy_rad = clus_sep[:,-1][-1]

# %%
# Clark data
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas_lib = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
clark = ascii.read(cata + 'clark_2018_.dat') 
ra_clark = []
dec_clark =[] 
for i in range(len(clark)):
    ra_i = clark['RAh','RAm','RAs'][i]
    dec_i = clark['DEd','DEm','DEs'][i]
    ra_clark.append('%sh%sm%s'%(ra_i[0],ra_i[1],ra_i[2]))
    dec_clark.append('-%sd%sm%s'%(dec_i[0],dec_i[1],dec_i[2]))

clark_coord = SkyCoord(ra = ra_clark, dec =dec_clark, unit = 'degree',obstime ='J2000',frame = 'fk5')

np_clark = np.c_[clark_coord.ra.value,clark_coord.dec.value,clark.columns['F127Mmag'].value,clark.columns['F153Mmag'].value]
np_clark_radec =  np.c_[clark_coord.ra.value,clark_coord.dec.value]
# Astroaling Hosek and Clark.
# I have selected four matchin stars by eye using Aladin adn use then for the astroalignemnt
hos_4 = np.loadtxt(pruebas_lib +'hos_four.txt')
cla_4 = np.loadtxt(pruebas_lib + 'clark_four.txt')
hos_radec = np.c_[racl.value,decl.value]
# m,(_,_)= aa.find_transform(np_clark_radec,hos_radec,max_control_points=100)
m,(_,_)= aa.find_transform(cla_4[:,0:2],hos_4[:,0:2],max_control_points=100)

print("Translation: (x, y) = (%.2f, %.2f)"%(m.translation[0],m.translation[1]))
print("Rotation: %.3f degrees"%(m.rotation ))
print("Scale factor: %.4f"%(m.scale))

test_trans = aa.matrix_transform(np_clark_radec, m.params)

np_clark[:,0] = test_trans[:,0]
np_clark[:,1] = test_trans[:,1]
np.savetxt(pruebas_lib + 'aa_clark.txt',np_clark,fmt ='%.8f '*2 +'%.3f '*2 ,header = 'Ra_aa Dec_aa F127M F153M')


# %%
if choosen_cluster == 'Arches':
    hl_lit = 12.5
else:
    hl_lit = 25
circle1 = plt.Circle((np.mean(racl.value), np.mean(decl.value)), hl_lit/3600, color='red',zorder =3, fill=False,
                     linewidth = 5, ls = 'dashed', label = 'hl-rad. = %s"'%(hl_lit))

ax[1].set_title('%s'%(choosen_cluster),fontsize = 30)
ax[0].scatter(arches['pmRA'], arches['pmDE'], alpha = alph, color = bg_cl)
ax[0].scatter(pmra, pmdec,edgecolor = 'k',s=100,alpha = 1)
ax[1].scatter(RA, DEC,alpha = alph, color = bg_cl)
ax[1].scatter(racl, decl,s=100,alpha = 1, edgecolor = 'k',)
ax[1].scatter(clark_coord.ra,clark_coord.dec,color ='fuchsia')
ax[1].add_patch(circle1)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2].scatter(arches['F127M'] - arches['F153M'],arches['F153M'],alpha = alph, color = bg_cl)
ax[2].scatter(color, f153,edgecolor = 'k',zorder=3,s=100,alpha = 1)
# hl_lit = 12.5 




vel_txt = '\n'.join(('$\\overline{\\mu}_{ra}$ = %.2f, $\\overline{\\mu}_{dec}$ = %.2f'%(np.mean(pmra),np.mean(pmdec)),
                     '$\\sigma_{\\mu ra}$ = %.2f, $\\sigma_{\\mu dec}$ = %.2f'%(np.std(pmra),np.std(pmdec))))   
propiedades = dict(boxstyle='round', facecolor= '#1f77b4', alpha=0.2)

ax[0].text(0.15, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
    verticalalignment='top', bbox=propiedades)

# ax[1].text(0.05, 0.95, 'hl radius $\\sim$ %.2f"\n # stars = %s '%(eff_rad,len(pmra)), transform=ax[1].transAxes, fontsize=30,
#                         verticalalignment='top', bbox=propiedades)
ax[1].text(0.05, 0.95, '# stars = %s '%(len(pmra)), transform=ax[1].transAxes, fontsize=30,
                        verticalalignment='top', bbox=propiedades)


txt_color = '\n'.join(('$\\overline{f127m-f153m}$ = %.2f'%(np.median(f127-f153)),
                                        '$\\sigma_{(f127m-f153m)}$ = %.2f'%(np.std(f127-f153)),
                                        '$\\Delta (f127m-f153m)$ = %.2f'%(max(f127-f153)-min(f127-f153))))
ax[2].text(0.35, 0.95, txt_color, transform=ax[2].transAxes, fontsize=25,
                        verticalalignment='top', bbox=propiedades)



ax[0].invert_xaxis()
ax[2].invert_yaxis()
ax[1].set_xlim(min(RA.value),max(RA.value))
ax[1].set_ylim(min(DEC.value),max(DEC.value))
ax[0].set_xlabel('$\\mu_{ra*}$ (mas yr$^{-1})$',fontsize =30) 
ax[0].set_ylabel('$\\mu_{dec}$ (mas yr$^{-1})$',fontsize =30)
if choosen_cluster == 'Arches': 
    ax[1].set_yticks([-28.84, -28.83, -28.81, -28.8])
if choosen_cluster == 'Quintuplet': 
    ax[1].set_yticks([-28.85, -28.84, -28.82, -28.81])
ax[1].set_xlabel('Ra (°)',fontsize =30) 
ax[1].set_ylabel('Dec (°)',fontsize =30)
ax[1].yaxis.set_label_coords(-0.05, 0.5)
ax[2].set_xlabel('f127m$-$f153m',fontsize =30) 
ax[2].set_ylabel('f153m',fontsize =30) 
ax[2].set_xlim(1.5, 5)
ax[1].legend(loc = 4)
# plt.savefig('/Users/amartinez/Desktop/PhD/My_papers/Libralato/%s_prob.png'%(choosen_cluster),dpi =300,bbox_inches = 'tight')
# sys.exit('113')
# %%
# Comparing with DBSCAN obtained cluster

# %%
# Dereddenig

# gns_pd = pd.read_fwf(gns_ext + '%s_gns.txt'%(choosen_cluster))
# gns = gns_pd.to_numpy()
# gns_coord = SkyCoord(ra=gns[:,1]*u.degree, dec=gns[:,3]*u.degree,frame ='icrs', equinox = 'J2000', obstime = 'J2015.5')
# idx = RA_DEC_clus.match_to_catalog_sky(gns_coord)
# validas = np.where(idx[1]<0.08*u.arcsec)

# Ks, AKs = gns[:,7][idx[0][validas]], gns[:,19][idx[0][validas]]
# for val in range(len(AKs)):
#     if AKs[val] !='---' and AKs[val] !='-1.00000':
#         AKs[val] = float(AKs[val])
#     else:
#         AKs[val] = 1  
# RA_DEC_clus =RA_DEC_clus[validas]
# pmra, pmdec = pmra[validas], pmdec[validas]
# racl, decl = racl[validas], decl[validas]
# # f127, f153 = f127[validas], f153[validas]-AKs
# good_Ks = np.where(Ks <90)
# # Ks = Ks[good_Ks]
# # AKs = AKs[good_Ks]
# f153 = (Ks-AKs)
# %%
# Half-light radio
species.SpeciesInit()   
synphot = species.SyntheticPhotometry('HST/WFC3_IR.F153M')
# synphot = species.SyntheticPhotometry('Paranal/HAWKI.Ks')

cent_sep = center_arc.separation(RA_DEC_clus).arcsec
flux = np.array([synphot.magnitude_to_flux(f153[mag], error=0.0, zp_flux=None)[0] for mag in range(len(f153))])
clus_sep =np.c_[racl.value,decl.value,f153.value,flux,cent_sep]
# clus_sep =np.c_[racl.value,decl.value,f153,flux,cent_sep]
light = sum(flux)

clus_sep = clus_sep[clus_sep[:, -1].argsort()]
cum = np.cumsum(clus_sep[:,3])
norm_cum = cum/light
hl_ind = np.where(cum < light/2)
eff_rad = clus_sep[hl_ind[0][-1]][-1]
# %%
fig, ax = plt.subplots(1,1)
ax.set_title('%s'%(choosen_cluster))
ax.scatter(clus_sep[:,-1], cum)
ax.axvline(eff_rad, label = 'hl_rad = %.2f'%(eff_rad))
ax.legend()

