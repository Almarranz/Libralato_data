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
import sys
# %%
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
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
#ra, dec, ID(in ACSWFC_PM or WFC3IR_PM),Original list, Altervative Id
yso_ra,yso_dec,yso_ID=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))





valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
catal=catal[valid]
# center=np.where(catal[:,12]-catal[:,14]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]

# mul_mc,mub_mc,dmul_mc,dmub_mc
gal_coor=catal[:,[17,18,19,20]]#this separation of the galactic pms itsnt really neccesary. It is a reminisce of the previous version of the script
# %%
#Selecting search radio and trasforming it to arcsec for naming differents lists
#Im doing this in order to compare the clusters found in different groups and check how much they depend of the searching radio 
radio=0.006
r_u=radio*u.degree
print('this is the searching radio: %s'%(r_u).to('arcsec'))
r_u=round(r_u.to('arcsec').value)


found=0
missing=0
# pms=[-3.156,-5.585,-6.411,-0.219]#this are the ecu(mua,mud) and galactic(mul,mub) pm of SrgA* (Reid & Brunthaler (2020))
pms=[0,0,0,0]
# pms=[0,0,-5.60,0.20] #this is from the dynesty adjustment
pms=np.array(pms)

for file_to_remove in glob.glob(pruebas+'group_radio%s*'%(r_u)):#Remove the files for previpus runs adn radios
    os.remove(file_to_remove) 
with open(pruebas+ 'MS_%s_.reg'%(name), 'w') as f:
        f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
        f.close
        
with open(pruebas+ 'pm_of_Ms_in_%s.txt'%(name), 'w') as f:
        f.write('#mul, mub, mua, mud, ra, dec,dmul,dmub,l,b, Ks, H, m139, position in GALCEN_TABLE_D.cat ')
        f.close
for i in range(len(yso_ra)):
# for i in range(1):    
    print(yso_ra[i],yso_dec[i])
    index=np.where((catal[:,5]==yso_ra[i]) & (catal[:,6]==yso_dec[i]) ) # looping a picking the stars coord on the Ms catalog
    if len(index[0]>0): 
        print(index[0])
        print(float(catal[index[0],5]),catal[index[0],6])
        with open(pruebas+ 'MS_%s_.reg'%(name), 'a') as f:
            f.write("\n"+'point(%s,%s) # point=x'%(float(catal[index[0],0]),float(catal[index[0],1]))+"\n"+
                    "\n"+ 'circle(%s,%s,%s)'%(float(catal[index[0],0]),float(catal[index[0],1]),radio)+' #text={%s,%s}'%(i,tipo[i]))
            f.close
        group=np.where(np.sqrt((catal[:,5]-catal[index[0],5])**2 + (catal[:,6]-catal[index[0],6])**2)< radio)
        print(len(group[0]))
        ra_=catal[:,5]
        dec_=catal[:,6]
        # Process needed for the trasnformation to galactic coordinates
        c = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree, frame='fk5')#you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames
        gal_c=c.galactic
        
        t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))
        
        df_gal=t_gal.to_pandas()
        gal=df_gal.to_numpy()
        formato='%.7f %.7f %.4f %.4f %.4f %.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.0f %.0f %.0f %.0f %.5f %.5f %.5f %.5f %.5f %.3f'
        # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
        #group_radio%s_%s_%s.txt are the stars around the Massive one from the list of massive stars, thar are also in the trimmed(or not) Libralato data
        np.savetxt(pruebas+'group_radio%s_%s_%s.txt'%(r_u,i,name),catal[group],fmt=formato,header=("'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'"))
        #This plots pmb vs pml and b vs l
# =============================================================================
#         fig, ax = plt.subplots(1,2,figsize=(20,10))
#        
#         # This will plot the vectors and stars in the galactic frame
#         t_gal['l'] = t_gal['l'].wrap_at('180d')#doesnt split the plot when the grpu fall both ways of l,b=0,0
#         ax[0].scatter(t_gal['l'][index[0]],t_gal['b'][index[0]],color='red',s=100)
#         ax[0].scatter(t_gal['l'][group[0]],t_gal['b'][group[0]])
#         ax[0].quiver([t_gal['l'][group[0]]],[t_gal['b'][group[0]]],np.array([gal_coor[group[0],0]])-pms[2],np.array([gal_coor[group[0],1]])-pms[3])
#         ax[0].set_xlabel(r'$\mathrm{l}$') 
#         ax[0].set_ylabel(r'$\mathrm{b}$') 
# 
# 
#         ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#         ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#         ax[0].legend(['yso #%s, %s, #stars=%s'%(i,tipo[i],len(gal[group[0],0]))],markerscale=1,loc=1,handlelength=1)
#         
#         
#         #group_%s_%s.txt are the stars around the Massive one from the list of massive stars, thar are also in the trimmed(or not) Libralato data
#         
#         ax[1].scatter([gal_coor[index[0],0]]-pms[2],[gal_coor[index[0],1]]-pms[3],color='red',s=100)
#         ax[1].scatter([gal_coor[group[0],0]]-pms[2],[gal_coor[group[0],1]]-pms[3], alpha =0.2)
#         
#         
#         
#         
#         ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#         ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#         ax[1].legend(['yso #%s, %s'%(i,tipo[i])],markerscale=1,loc=1,handlelength=1)
#         ax[1].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
#         ax[1].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
#         ax[1].invert_xaxis()
#         
#         # ax[1].axvline(pms[2], color='orange',linestyle='dashed', linewidth=1)
#         # ax[1].axhline(pms[3], color='orange',linestyle='dashed', linewidth=1)
#         ax[1].scatter(pms[2],pms[3],s=150, marker='*')
# =============================================================================
        found +=1
        t_gal['l'] = t_gal['l'].wrap_at('180d')#doesnt split the plot when the grpu fall both ways of l,b=0,0
        with open(pruebas+ 'pm_of_Ms_in_%s.txt'%(name), 'a') as f:#mul, mub, mua, mud, ra, dec,dmul,dmub,l,b, Ks, H,m139 position in GALCEN_TABLE_D.cat 
            f.write('\n'+ '%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.4f %.4f %.4f %.0f'%(float(gal_coor[index[0],0]),float(gal_coor[index[0],1]),float(catal[index[0],9]),
                                                                          float(catal[index[0],11]),float(catal[index[0],5]),float(catal[index[0],6]),float(catal[index[0],-4]),float(catal[index[0],-3]),t_gal['l'][index[0]].value,t_gal['b'][index[0]].value,
                                                                          float(catal[index[0],4]),float(catal[index[0],3]),float(catal[index[0],-2]),i))
                                                         # "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",

            f.close
       
    else:
        print('No mach in %s catalog'%(name))
        missing +=1
    # plt.xlabel(r'$\mathrm{\mu_{a}cosb (mas\ yr^{-1})}$') 
    # plt.ylabel(r'$\mathrm{\mu_{d} (mas\ yr^{-1})}$') 
    
print('Found %s , missing %s'%(found, missing))






