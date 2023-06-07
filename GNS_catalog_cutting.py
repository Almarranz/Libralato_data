#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:24:06 2023

@author: amartinez
"""

# We ara going to cut the gns catalog around the Arches or the Quituplet becouse
# the whole catalog is too big and it is a pain to work with
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from astropy.io import ascii
import astropy.coordinates as ap_coor
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
import pandas as pd

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
# '_RAJ2000', '_DEJ2000', 'RAJ2000', 'e_RAJ2000', 'DEJ2000', 'e_DEJ2000',
# 'RAJdeg', 'e_RAJdeg', 'DEJdeg', 'e_DEJdeg', 'RAHdeg', 'e_RAHdeg',
# 'DEHdeg', 'e_DEHdeg', 'RAKsdeg', 'e_RAKsdeg', 'DEKsdeg', 'e_DEKsdeg',
# 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Ksmag', 'e_Ksmag', 'iJ', 'iH',
# 'iKs'
gns_pd = pd.read_csv(cata + 'GNS_central.csv')# tCentral region of GNS
# %
gns = gns_pd.to_numpy()
# %%
gns_coord = SkyCoord(ra = gns[:,2], dec = gns[:,4],unit = 'degree')
m_arc = SkyCoord(ra = [266.46081294], dec = [-28.82303557],unit = 'degree' )
idxc, around_arc, d2d,d3d =  ap_coor.search_around_sky(m_arc, gns_coord, 80*u.arcsec)
gns_ar = gns[around_arc]
np.savetxt(cata +'GNS_arches.txt', gns_ar, fmt = '%.8f', header = '_RAJ2000 _DEJ2000 RAJ2000 e_RAJ2000 DEJ2000 e_DEJ2000 RAJdeg e_RAJdeg DEJdeg e_DEJdeg RAHdeg e_RAHdeg DEHdeg e_DEHdeg RAKsdeg e_RAKsdeg DEKsdeg e_DEKsdeg mag e_Jmag Hmag e_Hmag Ksmag e_Ksmag iJ iH iKs')








