#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:15:21 2022

@author: amartinez
"""

import numpy as np
from astropy.coordinates import match_coordinates_sky
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'

# name='ACSWFC'
name='WFC3IR'
trimmed_data = 'yes'
if trimmed_data == 'yes':
    pre=''
elif trimmed_data == 'no':
    pre='relaxed_'    

# not sure yet wich coordinates should I used for the matching. Ask paco why so many.
# ['_RAJ2000'0, '_DEJ2000'1, 'RAJ2000'2, 'e_RAJ2000'3, 'DEJ2000'4, 'e_DEJ2000'5, 'RAJdeg'6, 
#  'e_RAJdeg'7, 'DEJdeg'8, 'e_DEJdeg'9, 'RAHdeg'10, 'e_RAHdeg'11, 'DEHdeg'12, 
#  'e_DEHdeg'13, 'RAKsdeg'14, 'e_RAKsdeg'15, 'DEKsdeg'16, 'e_DEKsdeg'17, 
#  'Jmag'18, 'e_Jmag'19, 'Hmag'20, 'e_Hmag'21, 'Ksmag'22, 'e_Ksmag'23, 'iJ'24, 'iH'25, 'iKs'26]
# 27 columns
gns= pd.read_csv(cata + 'GNS_central.csv')# tCentral region of GNS

# 'ra dec x_c  y_c mua dmua mud dmud  time  n1  n2 ID mul mub dmul dmub mF139')
libr = np.loadtxt(results + '%srefined_%s_PM.txt'%(pre, name))
gns_np= gns.to_numpy()
# %%
gns_coord = SkyCoord(ra=gns_np[:,0]*u.degree, dec=gns_np[:,1]*u.degree)
libr_coord =  SkyCoord(ra=libr[:,0]*u.degree, dec=libr[:,1]*u.degree)
# %%
idx = libr_coord.match_to_catalog_sky(gns_coord)

# %% I cosider a math if the stars are less than 1 arcsec away 
valid = np.where(idx[1]<1*u.arcsec)

libr_match=libr[valid]
gns_match=gns_np[idx[0][valid]]
# %%Testing the macth
# =============================================================================
# gns_test = SkyCoord(ra=libr_march[:,0]*u.degree, dec=libr_march[:,1]*u.degree)
# libr_test =  SkyCoord(ra=gns_match[:,0]*u.degree, dec=gns_match[:,1]*u.degree)
# idx_test,d2d_test,d3d_test = libr_test.match_to_catalog_sky(gns_test)
# 
# print(max(d2d_test))
# 
# =============================================================================

# %%
# df = pd.read_csv(pruebas+'match_GNS_and_%s_refined.txt'%(name),sep=','
#                  ,names=['RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation'])

# catal_df = pd.read_csv(results+'%s_refined_with GNS_partner_mag_K_H.txt'%(name),
#                        sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'])

# not sure yet wich coordinates should I used for the matching. Ask paco why so many.
# ['_RAJ2000'0, '_DEJ2000'1, 'RAJ2000'2, 'e_RAJ2000'3, 'DEJ2000'4, 'e_DEJ2000'5, 'RAJdeg'6, 
#  'e_RAJdeg'7, 'DEJdeg'8, 'e_DEJdeg'9, 'RAHdeg'10, 'e_RAHdeg'11, 'DEHdeg'12, 
#  'e_DEHdeg'13, 'RAKsdeg'14, 'e_RAKsdeg'15, 'DEKsdeg'16, 'e_DEKsdeg'17, 
#  'Jmag'18, 'e_Jmag'19, 'Hmag'20, 'e_Hmag'21, 'Ksmag'22, 'e_Ksmag'23, 'iJ'24, 'iH'25, 'iKs'26]
gns_and_lib=np.c_[gns_match[:,[0,1,18,20,22]],libr_match, idx[1][valid].to(u.arcsec).value]

np.savetxt(pruebas + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name),gns_and_lib,
            header = 'RA_gns DE_gns Jmag Hmag Ksmag ra dec x_c y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub m139 Separation',
            fmt='%.7f %.7f %.4f %.4f %.4f %.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.0f %.0f %.0f %.0f %.5f %.5f %.5f %.5f %.5f %.3f')
gns_and_lib=np.c_[gns_match[:,[0,1,18,20,22]],libr_match]


# %%
print(np.std(idx[1][valid].to(u.arcsec).value))






