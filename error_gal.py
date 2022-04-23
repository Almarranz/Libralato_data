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

np.savetxt(cata+'GALCEN_%s_PM_galactic_skycoord.txt'%(name),np.array([mul,mub,dmul,dmub]).T,fmt='%.7f',header='mul, mub, dmul, dmub')











