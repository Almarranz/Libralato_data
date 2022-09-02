#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:45:22 2022

@author: amartinez
"""
# =============================================================================
# This script save the data for each subsection in txt files in order to be later
# used by simulated_cluster.pro
# =============================================================================
# %%imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
import os
import math
import shutil

# %%plotting parametres
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
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
carpeta = '/Users/amartinez/Desktop/PhD/Libralato_data/regions_for_simulations/'
name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

# %%
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
section = 'D'#selecting the whole thing

MS_ra,MS_dec = np.loadtxt(cata + 'MS_section%s.txt'%(section),unpack=True, usecols=(0,1),skiprows=0)
MS_coord = SkyCoord(ra = MS_ra*u.deg, dec = MS_dec*u.deg, frame = 'icrs',equinox ='J2000',obstime = 'J2014.2')
if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%
# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
# catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))


# %%
color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)


if section == 'A':
    m1 = -0.80
    m = 1
    step = 3300
    fila =-1
    lim_pos_up, lim_pos_down = 26700, 18500 #intersection of the positives slopes lines with y axis,
    lim_neg_up, lim_neg_down =32700,26000 #intersection of the negayives slopes lines with y axis,
    
    
    dist_pos = abs((-1*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))
    
    dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
    ang = math.degrees(np.arctan(m1))
    
    
    x_box_lst = [1,2,3]
    # samples_lst =[10,9,8,7,6,5]
    # x_box_lst = [3]
    samples_lst =[7]
    for x_box in x_box_lst:
        step = dist_pos /x_box
        step_neg =dist_neg/x_box
        
        for samples_dist in samples_lst:
           
            for ic in range(x_box*2-1):
                
                
                ic *= 0.5
                yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                # yg_2 = (lim_pos_up - (ic+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                yg_2 = (lim_pos_up - (ic+1)*step/np.cos(45*u.deg)) +  m*catal[:,7]
            
                # ax.plot(catal[:,7],yg_1, color ='g')
                # ax.plot(catal[:,7],yg_2, color ='g')
                for jr in range(x_box*2-1):
                    fig, ax = plt.subplots(1,1, figsize=(10,10))
                    ax.scatter(catal[:,0],catal[:,1],alpha =0.01)
                    jr *=0.5
                    yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                    # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                    yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                    good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                            & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
                    area = step*step_neg*0.05**2/3600
                    
                    
                    ax.scatter(catal[:,0][good],catal[:,1][good],color =strin[np.random.choice(indices)],alpha = 0.1)
                    
                    # ax.plot(catal[:,7],yr_1, color ='r')
                    # ax.plot(catal[:,7],yr_2, color ='r')
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    # place a text box in upper left in axes coords
                    txt ='central box ~ %.1f arcmin$^{2}$'%(area)
                    ax.text(0.65, 0.95, txt, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    ax.set_xlabel('Ra')
                    ax.set_ylabel('Dec')
                    ax.set_title('%.0f %.0f'%(ic*2,jr*2))
                    plt.show()
                    np.savetxt(carpeta + 'secA_area%.1f_%.0f_%.0f_dmu%s.txt'%(area,ic*2,jr*2,dmu_lim),catal[good], fmt = '%.6f',
                                    header = "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
     
if section == 'B':
    m1 = -0.80
    m = 1
    step = 3300
    lim_pos_up, lim_pos_down = 22000, -1000 #intersection of the positives slopes lines with y axis,
    lim_neg_up, lim_neg_down = 39000,31500 #intersection of the negayives slopes lines with y axis,
    
    dist_pos = abs((-1*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))
    dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
    # ang = math.degrees(np.arctan(m1))
    ang = (np.arctan(m1))
 
    xy_box_lst = [[3,1],[4,2],[6,2]]
    samples_lst =[10]
    
    # xy_box_lst = [[3,1]]
    # samples_lst =[10, 7]
    for elegant_loop in range(len(xy_box_lst)):
        x_box_lst = [xy_box_lst[elegant_loop][0]]
        y_box_lst = [xy_box_lst[elegant_loop][1]]
        for x_box in x_box_lst:
            step = dist_pos /x_box
            for y_box in y_box_lst:
                for samples_dist in samples_lst:
                    for ic in range(x_box*2-1):
                        
                        # fig, ax = plt.subplots(1,1,figsize=(10,10))
                        # ax.scatter(catal[:,7],catal[:,8],alpha =0.01)
                        ic *= 0.5
                        # yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                        yg_1 = (lim_pos_up - (ic)*step/np.cos(np.pi/4)) +  m*catal[:,7]
                        yg_2 = (lim_pos_up - (ic+1)*step/np.cos(np.pi/4)) +  m*catal[:,7]
                        
                        
                        for jr in range(y_box*2-1):
                            fig, ax = plt.subplots(1,1, figsize=(10,10))
                            step_neg =dist_neg/y_box
                            ax.scatter(catal[:,0],catal[:,1],alpha =0.01)
                            jr *=0.5
                            yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang)) +  m1*catal[:,7]
                            # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                            yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang)) +  m1*catal[:,7]
                            good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                                    & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
                            area = step*step_neg*0.05**2/3600
                            
                            
                            ax.scatter(catal[:,0][good],catal[:,1][good],color =strin[np.random.choice(indices)],alpha = 0.1)
                            
                            # ax.plot(catal[:,7],yr_1, color ='r')
                            # ax.plot(catal[:,7],yr_2, color ='r')
                            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                            # place a text box in upper left in axes coords
                            txt ='central box ~ %.1f arcmin$^{2}$'%(area)
                            ax.text(0.65, 0.95, txt, transform=ax.transAxes, fontsize=14,
                                verticalalignment='top', bbox=props)
                            ax.set_xlabel('Ra')
                            ax.set_ylabel('Dec')
                            ax.set_title('%.0f %.0f'%(ic*2,jr*2))
                            plt.show()
                            np.savetxt(carpeta + 'secB_area%.1f_%.0f_%.0f_dmu%s.txt'%(area,ic*2,jr*2,dmu_lim),catal[good], fmt = '%.6f',
                                           header = "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
if section == 'C':
    m1 = -0.80
    m = 1
    step = 3300


    #This for removin previous subsections
    # for f_remove in glob.glob(pruebas + 'subsec_%s/subsec*'%(section)):
    #     os.remove(f_remove)

    missing =0
    # fig, ax = plt.subplots(1,1, figsize=(10,10))
    # ax.scatter(catal[:,7],catal[:,8])
    fila =-1
    lim_pos_up, lim_pos_down = 1000, -18000 #intersection of the positives slopes lines with y axis,
    lim_neg_up, lim_neg_down =30500,22500 #intersection of the negayives slopes lines with y axis,

    
    dist_pos = abs((-1*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))

    dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
    ang = math.degrees(np.arctan(m1))

    clus_num = 0
    # x_box = 3




    clustered_by_list =['all_color']
    xy_box_lst = [[3,1],[4,2],[6,2]]
    # xy_box_lst = [[4,2]]

    samples_lst =[10]
    # samples_lst =[7]

    for a in range(len(clustered_by_list)):
        clustered_by = clustered_by_list[a]
        # %
        for elegant_loop in range(len(xy_box_lst)):
            x_box_lst = [xy_box_lst[elegant_loop][0]]
            y_box_lst = [xy_box_lst[elegant_loop][1]]
            for x_box in x_box_lst:
                step = dist_pos /x_box
                for y_box in y_box_lst:
                    for samples_dist in samples_lst:
                        for ic in range(x_box*2-1):
                            
                            # fig, ax = plt.subplots(1,1,figsize=(10,10))
                            # ax.scatter(catal[:,0],catal[:,1],alpha =0.01)
                            ic *= 0.5
                            yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                            # yg_2 = (lim_pos_up - (ic+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                            yg_2 = (lim_pos_up - (ic+1)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                            
                           
                            for jr in range(y_box*2-1):
                                fig, ax = plt.subplots(1,1, figsize=(10,10))
                                step_neg =dist_neg/y_box
                                ax.scatter(catal[:,0],catal[:,1],alpha =0.01)
                                jr *=0.5
                                yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                                # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                                yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                                good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                                        & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
                                area = step*step_neg*0.05**2/3600
                                
                                
                                ax.scatter(catal[:,0][good],catal[:,1][good],color =strin[np.random.choice(indices)],alpha = 0.1)
                                
                                
                                props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                                # place a text box in upper left in axes coords
                                txt ='central box ~ %.1f arcmin$^{2}$'%(area)
                                ax.text(0.65, 0.95, txt, transform=ax.transAxes, fontsize=14,
                                    verticalalignment='top', bbox=props)
                                ax.set_xlabel('Ra')
                                ax.set_ylabel('Dec')
                                ax.set_title('%.0f %.0f'%(ic*2,jr*2))
                                plt.show()
                                np.savetxt(carpeta + 'secC_area%.1f_%.0f_%.0f_dmu%s.txt'%(area,ic*2,jr*2,dmu_lim),catal[good], fmt = '%.6f',
                                               header = "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
if section == 'D':
    m1 = -0.80
    m = 1
    step = 3300
    

    lim_pos_up, lim_pos_down = 500, -18000 #intersection of the positives slopes lines with y axis,
    lim_neg_up, lim_neg_down =23000,16500 #intersection of the negayives slopes lines with y axis,
    
   
    dist_pos = abs((-1*catal[0,7]+ (lim_pos_down + m*catal[0,7])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))
    
    
    dist_neg = abs((-m1*catal[0,7]+ (lim_neg_down + m1*catal[0,7])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
    ang = math.degrees(np.arctan(m1))
    
    
    
    
    
    xy_box_lst = [[3,1],[4,2],[6,2]]
    # xy_box_lst = [[4,2]]
    
    samples_lst =[10]
    # samples_lst =[7]
    
    
    for elegant_loop in range(len(xy_box_lst)):
        x_box_lst = [xy_box_lst[elegant_loop][0]]
        y_box_lst = [xy_box_lst[elegant_loop][1]]
        for x_box in x_box_lst:
            step = dist_pos /x_box
            for y_box in y_box_lst:
                for samples_dist in samples_lst:
                    for ic in range(x_box*2-1):
                        
                        # fig, ax = plt.subplots(1,1,figsize=(10,10))
                        # ax.scatter(catal[:,7],catal[:,8],alpha =0.01)
                        ic *= 0.5
                        yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                        # yg_2 = (lim_pos_up - (ic+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                        yg_2 = (lim_pos_up - (ic+1)*step/np.cos(45*u.deg)) +  m*catal[:,7]
                        
                        # ax.scatter(catal[:,7][good],catal[:,8][good],color =strin[np.random.choice(indices)],alpha = 0.1)
            
             # %       
                        
                        for jr in range(y_box*2-1):
                            fig, ax = plt.subplots(1,1, figsize=(10,10))
                            step_neg =dist_neg/y_box
                            ax.scatter(catal[:,0],catal[:,1],alpha =0.01)
                            jr *=0.5
                            yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                            # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                            yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang*u.deg)) +  m1*catal[:,7]
                            good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                                                    & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
                            area = step*step_neg*0.05**2/3600
                            
                            
                            ax.scatter(catal[:,0][good],catal[:,1][good],color =strin[np.random.choice(indices)],alpha = 0.1)
                            
                           
                            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                            # place a text box in upper left in axes coords
                            txt ='central box ~ %.1f arcmin$^{2}$'%(area)
                            ax.text(0.65, 0.95, txt, transform=ax.transAxes, fontsize=14,
                                verticalalignment='top', bbox=props)
                            ax.set_xlabel('Ra')
                            ax.set_ylabel('Dec')
                            plt.show()
                            np.savetxt(carpeta + 'secD_area%.1f_%.0f_%.0f_dmu%s.txt'%(area,ic*2,jr*2,dmu_lim),catal[good], fmt = '%.6f',
                                           header = "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")





    
    
    
    











          
