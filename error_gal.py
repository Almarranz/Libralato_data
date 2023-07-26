#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:45:48 2022

@author: amartinez
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import sys
# =============================================================================
# # Transformation of pm and their uncertainties to galactic using SkyCoord for both.
# =============================================================================
cata = '/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results = '/Users/amartinez/Desktop/PhD/Libralato_data/results/'


# name = 'ACSWFC'
name = 'WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
catal = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
# %%
ra = catal[:,0]*u.degree
dec = catal[:,1]*u.degree
pmra = catal[:,4]*u.mas/u.yr
pmdec = catal[:,6]*u.mas/u.yr
dpmra = catal[:,5]*u.mas/u.yr
dpmdec = catal[:,7]*u.mas/u.yr



#%%
pm_ecu = SkyCoord(ra  = ra,
                        dec = dec,
                        pm_ra_cosdec = pmra,
                        pm_dec = pmdec,
                        frame = 'icrs')
pm_gal = pm_ecu.galactic
print('pm_gal ready')
# %%
dpm_ecu = SkyCoord(ra  = ra,
                        dec = dec,
                        pm_ra_cosdec = dpmra,
                        pm_dec = dpmdec,
                        frame = 'icrs')
dpm_gal = dpm_ecu.galactic
print('dpm_gal ready')

# %%
noventa=np.where(dpm_ecu.pm_ra_cosdec.value>90)

mul, mub = pm_gal.pm_l_cosb.value, pm_gal.pm_b.value
dmul, dmub = dpm_gal.pm_l_cosb.value, dpm_gal.pm_b.value



# %%
dmul[noventa] = 99.99
dmub[noventa]  = 99.99

# np.savetxt(pruebas+'erase_this.txt',np.array([mul_mc,mub_mc,dmul_mc,dmub_mc]).T)

# np.savetxt(cata+'GALCEN_%s_PM_galactic_skycoord.txt'%(name),np.array([mul,mub,dmul,dmub]).T,fmt='%.7f',header='mul, mub, dmul, dmub')



# %%
# Transformation for the SgrA* to galactic
# according to Xu, Zhang, Reid+2022 
# ra_sA = '17h45m40.032863s'
# dec_sA = '−29d00m28.24260s'
# pmra_sA = -3.152*u.mas/u.yr
# pmdec_sA = -5.586*u.mas/u.yr

# according to Gordon, witt, +2023
ra_sA = '17h45m40.034047s'
dec_sA = '−29°00′28.21601" '
pmra_sA = -3.128*u.mas/u.yr
pmdec_sA = -5.584*u.mas/u.yr

coor_sA = SkyCoord(ra = ra_sA, dec = dec_sA, 
                   pm_ra_cosdec = pmra_sA,
                   pm_dec = pmdec_sA,
                   frame = 'icrs').galactic
# %%
pml_sA = coor_sA.pm_l_cosb.value
pmb_sA = coor_sA.pm_b.value
pml_Lib_b = -5.74
pmb_Lib_E = -3.77
pmb_Lib_W = -7.91
neg =-1 
print('Bulge = %.2f'%(-pml_Lib_b+pml_sA))
print('East = %.2f'%(-pmb_Lib_E+pml_sA))
print('West = %.2f'%(-pmb_Lib_W+pml_sA))

print('Bulge = %.2f'%(pml_Lib_b-pml_sA))
print('East = %.2f'%(pmb_Lib_E-pml_sA))
print('West = %.2f'%(pmb_Lib_W-pml_sA))
# Comparing results with Libralato paper






