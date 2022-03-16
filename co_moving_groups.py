#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:02:37 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable
from matplotlib import rcParams
import os
import glob
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
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
#ra, dec, ID(in ACSWFC_PM or WFC3IR_PM),Original list, Altervative Id
yso_ra,yso_dec,yso_ID=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')
# yso_df=pd.read_csv(cata+'GALCEN_TABLE_D.cat', sep=' ')
# yso=yso_df.to_numpy()


# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
# catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
# catal=np.loadtxt(results+'refined_%s_PM.txt'%(name))
catal_df=pd.read_csv(pruebas+'relaxed_%s_refined_with_GNS_partner_mag_K_H.txt'%(name),sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','mul_mc','mub_mc','dmul_mc','dmub_mc','time','n1','n2','idt','m139','Separation','Ks','H'])

# mul_mc,mub_mc,dmul_mc,dmub_mc


catal=catal_df.to_numpy()
gal_coor=catal[:,8:12]#in the origial script the galactic velocities and uncertainties were in a different file. Im doing this so I dont have to change this script that much.
valid=np.where(np.isnan(catal[:,14])==False)
catal=catal[valid]
gal_coor=gal_coor[valid]
# no_fg=np.where(catal[:,12]-catal[:,14]>2.5)
no_fg=np.where(catal[:,-1]-catal[:,-2]>1.3)
catal=catal[no_fg]
gal_coor=gal_coor[no_fg]
catal=np.c_[catal,gal_coor[:,0],gal_coor[:,1]]#in here we add the values for the galactic pm NOT galactic coordinates
# %%
lim_dmul=0.5
#Should be a way to import a variable from another script, but this doesnt work
# from vx_gsan_fit import lim_dmul # this import the limit in velocity we used for fitting the distribution to the gaussians

accu=np.where((abs(gal_coor[:,2])<lim_dmul) & (abs(gal_coor[:,3])<lim_dmul))#Are they in the paper selecting by the error of the galactic or equatorial coordintes???
catal=catal[accu]
# %%
radio=0.009
found=0
missing=0
# pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
pms=[0,0,0,0]
# pms=[0,0,-5.60,0.20] #this is from the dynesty adjustment
pms=np.array(pms)
for file_to_remove in glob.glob(pruebas+'group_*'):
    os.remove(file_to_remove) 

# %%
with open(pruebas+ 'MS_%s_.reg'%(name), 'w') as f:
        f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
        f.close
        
with open(pruebas+ 'pm_of_Ms_in_%s.txt'%(name), 'w') as f:
        f.write('#mul, mub, mua, mud, ra, dec,x_c,y_c,l,b, Ks, H position in GALCEN_TABLE_D.cat ')
        f.close
for i in range(len(yso_ra)):
# for i in range(3,4):    
    print(yso_ra[i])
    index=np.where((catal[:,0]==yso_ra[i]) & (catal[:,1]==yso_dec[i]) )
    if len(index[0]>0):
        print(index[0])
        print(float(catal[index[0],0]),catal[index[0],1])
        with open(pruebas+ 'MS_%s_.reg'%(name), 'a') as f:
            f.write("\n"+'point(%s,%s) # point=x'%(float(catal[index[0],0]),float(catal[index[0],1]))+"\n"+
                    "\n"+ 'circle(%s,%s,%s)'%(float(catal[index[0],0]),float(catal[index[0],1]),radio)+' #text={%s,%s}'%(i,tipo[i]))
            f.close
        group=np.where(np.sqrt((catal[:,0]-catal[index[0],0])**2 + (catal[:,1]-catal[index[0],1])**2)< radio)
        print(len(group[0]))
        ra_=catal[:,0]
        dec_=catal[:,1]
        # Process needed for the trasnformation to galactic coordinates
        c = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree, frame='fk5')
        gal_c=c.galactic
        
        t_gal= QTable([gal_c], names=["lines coord"])
        
        df_gal=t_gal.to_pandas()
        gal=df_gal.to_numpy()
        
        fig, ax = plt.subplots(1,2,figsize=(20,10))
        # This will plot the vector and stars in the ecuatorial frame
# =============================================================================
#         ax[0].scatter(catal[index[0],0],catal[index[0],1],color='red',s=100)
#         ax[0].scatter(catal[group[0],0],catal[group[0],1])
#         # ax.quiver(catal[index[0],0],catal[index[0],1],[catal[index[0],4]],[catal[index[0],6]],alpha=0.2)#this is for the vector on the Ms object in ecuatorial
#         # ax.quiver(catal[index[0],0],catal[index[0],1],[gal_coor[index[0],0]],[gal_coor[index[0],1]])#this is for the vector on the Ms object in galactic
#         ax[0].quiver([catal[group[0],0]],[catal[group[0],1]],np.array([catal[group[0],4]])-pms[0],np.array([catal[group[0],6]])-pms[1],alpha=0.2)
#         ax[0].quiver([catal[group[0],0]],[catal[group[0],1]],np.array([gal_coor[group[0],0]])-pms[2],np.array([gal_coor[group[0],1]])-pms[3])
#         ax[0].set_xlabel(r'$\mathrm{ra}$') 
#         ax[0].set_ylabel(r'$\mathrm{dec}$') 
# 
# =============================================================================
        # This will plot the vectors and stars in the galactic frame

        ax[0].scatter(gal[index[0],0],gal[index[0],1],color='red',s=100)
        ax[0].scatter(gal[group[0],0],gal[group[0],1])
        # ax.quiver(catal[index[0],0],catal[index[0],1],[catal[index[0],4]],[catal[index[0],6]],alpha=0.2)#this is for the vector on the Ms object in ecuatorial
        # ax.quiver(catal[index[0],0],catal[index[0],1],[gal_coor[index[0],0]],[gal_coor[index[0],1]])#this is for the vector on the Ms object in galactic
        # ax[0].quiver([gal[group[0],0]],[gal[group[0],1]],np.array([catal[group[0],4]])-pms[0],np.array([catal[group[0],6]])-pms[1],alpha=0.2)
        ax[0].quiver([gal[group[0],0]],[gal[group[0],1]],np.array([gal_coor[group[0],0]])-pms[2],np.array([gal_coor[group[0],1]])-pms[3])
        ax[0].set_xlabel(r'$\mathrm{l}$') 
        ax[0].set_ylabel(r'$\mathrm{b}$') 


        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[0].legend(['yso #%s, %s, #stars=%s'%(i,tipo[i],len(gal[group[0],0]))],markerscale=1,loc=1,handlelength=1)
        
        
        np.savetxt(pruebas+'group_%s_%s.txt'%(i,name),np.c_[catal[group],gal[group][:,0],gal[group][:,1]],fmt='%.7f',header=('ra,dec,x_c,y_c,mua,dmua,mud,dmud,time,n1,n2,idt,m139,Separation,Ks,H,mul,mub,l,b'))
        
        ax[1].scatter([gal_coor[index[0],0]]-pms[2],[gal_coor[index[0],1]]-pms[3],color='red',s=100)
        ax[1].scatter([gal_coor[group[0],0]]-pms[2],[gal_coor[group[0],1]]-pms[3], alpha =0.2)
                                                                #mul, mub, mua, mud, ra, dec,x_c,y_c,l,b, Ks, H position in GALCEN_TABLE_D.cat
        with open(pruebas+ 'pm_of_Ms_in_%s.txt'%(name), 'a') as f:
            f.write('\n'+ '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.0f'%(float(gal_coor[index[0],0]),float(gal_coor[index[0],1]),float(catal[index[0],4]),
                                                                          float(catal[index[0],6]),float(catal[index[0],0]),float(catal[index[0],1]),float(catal[index[0],2]),float(catal[index[0],3]),float(gal[index[0],0]),float(gal[index[0],1])
                                                                          ,float(catal[index[0],-4]),float(catal[index[0],-3]),i))

            f.close
       
        
        
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax[1].legend(['yso #%s, %s'%(i,tipo[i])],markerscale=1,loc=1,handlelength=1)
        ax[1].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
        ax[1].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
        ax[1].invert_xaxis()
        
        # ax[1].axvline(pms[2], color='orange',linestyle='dashed', linewidth=1)
        # ax[1].axhline(pms[3], color='orange',linestyle='dashed', linewidth=1)
        ax[1].scatter(pms[2],pms[3],s=150, marker='*')
        found +=1
# =============================================================================
#         fig, ax = plt.subplots(1,1,figsize=(10,10))
#         ax.hist(catal[group[0],4],bins='auto') 
#         ax.hist(catal[group[0],6],alpha=0.5,bins='auto') 
#         ax.legend(['mua (yso #%s)'%(i),'mub'],markerscale=1,loc=1,handlelength=1)
# =============================================================================
        
    else:
        print('No mach in %s catalog'%(name))
        missing +=1
    # plt.xlabel(r'$\mathrm{\mu_{a}cosb (mas\ yr^{-1})}$') 
    # plt.ylabel(r'$\mathrm{\mu_{d} (mas\ yr^{-1})}$') 
    
print('Found %s , missing %s'%(found, missing))






