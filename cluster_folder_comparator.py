#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:40:34 2022

@author: amartinez
"""

# =============================================================================
# We are going to copare differents file in differents folders to finde commons
# elemts in the different  clusters.
# =============================================================================

import numpy as np
import os
import glob

pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
morralla =  '/Users/amartinez/Desktop/morralla/'

folder1 = '/Users/amartinez/Desktop/morralla/Sec_A_all_good_VvsL_mass/'
folder2 = '/Users/amartinez/Desktop/morralla/Sec_A_all_color_good_VvsL_mass/'
counter =0
for f1 in glob.glob(folder1 + 'cluster_num*'):
    # print(f1)
    # print(30*'$')
    clus_lst = os.listdir(f1)
    for f1_clus in clus_lst:
        # print(f1+'/'+f1_clus)
        ra_dec_clus =np.loadtxt(f1+'/'+f1_clus, usecols=(0,1))
        # print(30*'+')   
        for f2 in glob.glob(folder2 + 'cluster_num*'):
            # print(f2)
            clus_lst2 = os.listdir(f2)
            for f2_clus in clus_lst2:
                counter +=1
                # print(f1_clus,f2_clus,counter)
                ra_dec =np.loadtxt(f2+'/'+f2_clus, usecols=(0,1))
                aset = set([tuple(x) for x in ra_dec_clus])
                bset = set([tuple(x) for x in ra_dec])
                intersection = np.array([x for x in aset & bset])
                # intersection_lst.append(len(intersection))
                # print('This is intersection',intersection_lst)
                print(30*'-')
                print(f1_clus,f2_clus)
                print(30*'-')
                # print('Same (or similar) cluster  is in %s %s'%(ra_dec,ra_dec_clus))
                if len(intersection)> 0 :
                    print(30*'!!')
                    print('Same (or similar) cluster  is in %s %s'%(f1_clus,f2_clus))
                    print(30*'!!')

# %%

 
        
 
    
 
    
 
    
 
    
 
    
 
    
 
    