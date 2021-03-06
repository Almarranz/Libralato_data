#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:31:49 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt

# %%

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'

#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID
# name='ACSWFC'
name='WFC3IR'

# Pm catalog are not used in this scripts, do not know why is this here...
# =============================================================================
# # ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
# catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
# # ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt, mul, mub, dmul, dmub= np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
# catal=np.loadtxt(cata + '%s_pm_galactic.txt'%(name))# this is the Libralato`s catalog plus the galatic proper motions
# =============================================================================

# %%

# mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )
all_ep1=np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=False)
# mag2, rms2, qfit2, o2, RADXS2, nf2, nu2, Localsky2, Local_skyrms2= np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=True )
all_ep2=np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=False )

# %%
for epoch in range(1,3):#loop to avoid to run epch 1 and epch 2 separatly
# epoch=1
    if epoch == 1:
        ep1_test = all_ep1
        num=np.arange(0,len(all_ep1),1)
        ep1_test = np.c_[ep1_test,num]# Added an ID. Stars are in the same order in both lists
    elif epoch == 2:
        ep1_test = all_ep2
        num=np.arange(0,len(all_ep2),1)
        ep1_test=np.c_[ep1_test,num]# Added an ID. Stars are in the same order in both lists
    
    # %%
    paso=0.1
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    n,bins_edges,otro=ax.hist(ep1_test[:,0],bins=np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),paso),linewidth=2,edgecolor='black') 
    mag_b=np.digitize(ep1_test[:,0], np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),paso), right=True)
    # %%
    trim_data= 'no'
    # Condition (a): qfit > percentile85 in bins of 1mag width (the width of 1mag i guessed)
    if trim_data == 'yes':
        all_sum=[]
        qfit_valid=[]
        for i in range(min(mag_b),(max(mag_b)+1)):
            try:
                mag_binned=np.where(mag_b==i)
                qfit_i=ep1_test[:,2][mag_binned]
                print('%.5f'%(np.percentile(qfit_i,85)),i,len(qfit_i),len(mag_binned[0]))
                perc = np.percentile(qfit_i,85)
                for j in range(len(qfit_i)):
                    if qfit_i[j] > 0.6:
                        if qfit_i[j] >= perc or qfit_i[j] >= 0.975:
                            qfit_valid.append(mag_binned[0][j])
            except:
                print('Fallo:',i,len(qfit_i),len(mag_binned[0]))
                        
            all_sum.append(len(qfit_i))
        print(sum(all_sum),'All')
        print(len(qfit_valid))
        ep1_test=ep1_test[qfit_valid]
        print(len(ep1_test),'Condition a')
        # %% 
        # Condition (b): rms > percentile85 in bins of 1mag(the 1mag bins I guessed)
        all_sum=[]
        rms_valid=[]
        mag_b=np.digitize(ep1_test[:,0], np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),paso), right=True)
        
        # for i in range(len(np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),1))+1):#that last +1 if for picking up the last bin
        for i in range(min(mag_b),(max(mag_b)+1)):
            try:
                mag_binned=np.where(mag_b==i)
                rms_i=ep1_test[:,1][mag_binned]
                print('%.5f'%(np.percentile(rms_i,85)),i,len(rms_i),len(mag_binned[0]))
                perc = np.percentile(rms_i,85)
                all_sum.append(len(rms_i))
                for j in range(len(rms_i)):
                    if rms_i[j] < 0.5:
                        if (rms_i[j] <= 0.1) or (rms_i[j] <= perc):
                            rms_valid.append(mag_binned[0][j]) 
            except :
                print('Fallo:',len(rms_i))
            
        print(sum(all_sum),'All after a')
        print(len(rms_valid))
        ep1_test=ep1_test[rms_valid]
        print(len(ep1_test),'Condition b')
        # %%
        # Condition (c): ratio2=nu2/nf2 bigger than 50% (in the paper it says SMALLER but I think it is a typo)
        # mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms
        nu_good=np.where(ep1_test[:,5] > 0)
        ep1_test=ep1_test[nu_good]
        ratio=np.where(ep1_test[:,6]/ep1_test[:,5] >= 0.5)
        # ratio=np.where(ep1_test[:,6]/ep1_test[:,5] <= 0.5)
        
        ep1_test=ep1_test[ratio]
        print(len(ep1_test),'Condition c')
        # %%
        # Condition (d): 
        flux_good=np.where(ep1_test[:,3]<1)
        ep1_test=ep1_test[flux_good]
        print(len(ep1_test),'Condition d')
        #%%
        #Condition (e): RADXS is lower than the 85th percentile in bins of 1mag width
        
        final_index=[]
        all_sum=[]
        radxs_valid=[]
        mag_b=np.digitize(ep1_test[:,0], np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),paso), right=True)
        
        # for i in range(len(np.arange(np.round(min(ep1_test[:,0])),np.round(max(ep1_test[:,0])+1),1))+1):#that last +1 if for picking up the last bin
        for i in range(min(mag_b),(max(mag_b)+1)):
            try:
                mag_binned=np.where(mag_b==i)
                radxs_i=ep1_test[:,4][mag_binned]
                radxs_i=np.absolute(radxs_i)
                print('%.5f'%(np.percentile(radxs_i,85)),i,len(radxs_i),len(mag_binned[0]))
                perc = np.percentile(radxs_i,85)
                all_sum.append(len(radxs_i))
                for j in range(len(radxs_i)):
                    if radxs_i[j] < 0.1:
                        if (radxs_i[j] <= 0.01) or (radxs_i[j] <= perc):
                            radxs_valid.append(mag_binned[0][j])
            except:
                 print('Fallo:',len(radxs_i))
        print(sum(all_sum),'All after condiotion d')
        print(len(radxs_valid))
        ep1_test=ep1_test[radxs_valid]
        print(len(ep1_test),'Condition e')
        
        # %%
        # ,fmt='%.4f %.4f %.4f %.4f %.4f %.0f %.0f %.2f %.2f %.0f                                        
        np.savetxt(results+'foto_well_mesaured_ep%s_%s.txt'%(name,epoch),ep1_test,delimiter='   ',fmt='%.4f %.4f %.4f %.4f %.4f %.0f %.0f %.2f %.2f %.0f',header='index for the stars that fullfil the well_mesaured critreia from Libralato et al. 2021'+'\n'+'mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms, idt')
    
    elif trim_data=='no':
        np.savetxt(results+'relaxed_foto_well_mesaured_ep%s_%s.txt'%(name,epoch),ep1_test,delimiter='   ',fmt='%.4f %.4f %.4f %.4f %.4f %.0f %.0f %.2f %.2f %.0f',header='index for the stars that fullfil the well_mesaured critreia from Libralato et al. 2021'+'\n'+'mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms, idt')
    
    else:
        print('you have to set trim_data variable to either "yes" or "no"')















