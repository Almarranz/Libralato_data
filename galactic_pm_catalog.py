#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:24:50 2022

@author: amartinez
"""
import numpy as np
# import pandas as pd

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID
# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, ID = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
# mul_mc,mub_mc,dmul_mc,dmub_mc
catal_pm=np.loadtxt(cata + 'GALCEN_%s_PM_galactic.txt'%(name))

catal = np.c_[catal,catal_pm]

np.savetxt(cata + '%s_pm_galactic.txt'%(name),catal,delimiter=' ',
           fmt='%.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.4f %.0f %.0f %.0f %.5f %.5f %.5f %.5f',
           header = 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul_mc mub_mc dmul_mc dmub_mc ')

# %% This give you the catalog in a nice format, if you need it.
# =============================================================================
# df = pd.DataFrame(catal, columns=['ra','dec','x_c','y_c', 'mua', 'dmua', 'mud', 'dmud','time', 'n1', 'n2', 'idt', 'mul_mc', 'mub_mc', 'dmul_mc', 'dmub_mc'])
# df.to_csv(pruebas + "output.cvs")
# =============================================================================
