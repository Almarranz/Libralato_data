#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:18:11 2022

@author: amartinez
"""

# =============================================================================
# This will plot indiviual clusters (or cluster with common(s) stars in the same 
# folder) to be used in documents and such. Basically is a copy of candidates_plotting.py
# =============================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.coordinates as ap_coor
import pandas as pd
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
from astropy.stats import sigma_clip
from matplotlib.ticker import FormatStrFormatter
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
morralla =  '/Users/amartinez/Desktop/morralla/'
regions = '/Users/amartinez/Desktop/PhD/regions/Libralato/'

# %%
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
frase = 'On which section are you working?'
print('\n'.join((len(frase)*'π',frase+'\n A, B, C or D',len(frase)*'π')))
section = input('Awnser:')
# section = 'C'#selecting the whole thing

ban_cluster = np.loadtxt(cata +'ban_cluster.txt')
ban_coord = SkyCoord(ra = ban_cluster[:,0]*u.deg, dec = ban_cluster[:,1]*u.deg, frame ='icrs', equinox = 'J2000', obstime = 'J2015.5')
# print(ban_cluster[:,1])
# print(ban_coord.dec.value)
if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
    
# %%
gns= pd.read_csv(cata + 'GNS_central.csv')# tCentral region of GNS
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
#here we can save the trimmed catal for using it in Aladin
np.savetxt(pruebas + 'libralato_center_dpm%s.txt'%(dmu_lim),catal)
color_de_cluster = 'lime'

# for clus_f in glob.glob(folder +'*'):
# % coordinates
ra_=catal[:,5]
dec_=catal[:,6]
# Process needed for the trasnformation to galactic coordinates
coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree, frame ='icrs', equinox ='J2000', obstime = 'J2014.2')#
gal_c=coordenadas.galactic

t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))

mul,mub = catal[:,-6],catal[:,-5]
x,y = catal[:,7], catal[:,8]
colorines = catal[:,3]-catal[:,4]
# %%Thi is for the extinction

Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()#TODO
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
AKs_center =AKs_np[center]#TODO
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree,frame ='icrs', equinox ='J2000', obstime = 'J2015.5')
# %
# %
AKs_list1 =  np.arange(1.6,2.11,0.01)
AKs_list = np.append(AKs_list1,0)#I added the 0 for the isochrones without extiction
# %%
# This deletes the folder
sec_clus = morralla +'combined_Sec_%s_clus/'%(section)
ifE_sec = os.path.exists(sec_clus)
if not ifE_sec:
    os.makedirs(morralla+ 'combined_Sec_%s_clus/cluster_num/'%(section))

carp_clus = sec_clus +'/ban_cluster_num*'


clus_to_erase = glob.glob(morralla + 'combined_Sec_%s_clus/cluster_num/cluster_combined_clus*'%(section))
for f_e in range(len(clus_to_erase)):
    print(clus_to_erase[f_e])
    # shutil.rmtree(clus_to_erase[f_e], ignore_errors=True)
    os.remove((clus_to_erase[f_e]))
    
# %%
# choosen_folder = input('Copy paste the folder with the clusters:')
choosen_folder = 'Sec_B_dmu1_at_minimun_2022-08-30'
# section_folder = '/Users/amartinez/Desktop/morralla/Sec_%s_dmu2_at_minimun_2022-08-30/'%(section)#TODO
section_folder = '/Users/amartinez/Desktop/morralla/%s/'%(choosen_folder)#TODO

print(os.path.basename(section_folder))
# section_folder = '/Users/amartinez/Desktop/morralla/Sec_A_at_2022-07-20 12/'#Test folder
if 'dmu0.5' in section_folder:
    lim = '0.5'
elif 'dmu1' in section_folder:
    lim = '1.0'
elif 'dmu2' in section_folder:
    lim = '2.0'
elif 'dmu3' in section_folder:
    lim = '3.0'

plots =0
for folder in sorted(glob.glob(section_folder + 'cluster_num*'),key = os.path.getmtime):
   print(folder)
   for cluster_f in sorted(glob.glob(folder + '/cluster*.txt'),key = os.path.getmtime):
       print(cluster_f)
   plots +=1
   print(30*'+')
print(plots)
# %%
fs = 25
plots =0

# for folder in sorted(glob.glob(section_folder + 'cluster_num*'),key = os.path.getmtime):
for folder in sorted(glob.glob(section_folder + 'cluster_num32_2_knn10_area7.49'),key = os.path.getmtime):#cluster for testing
   
    all_clus = []
    cluster_len =[]
    equal_method=[]
    clus_per_folder =0
    for file in glob.glob(folder + '/cluster*'):
        if 'all_color.txt' in os.path.basename(file):
            equal_method.append(0)
        elif 'all.txt' in os.path.basename(file):
            equal_method.append(1)
        clus_per_folder += 1  
        # ra, dec, l, b, pml, pmb,J, H, Ks,x, y, AKs_mean, dAks_mean, radio("),cluster_ID
        cluster = np.loadtxt(file)
        cluster_len.append(len(cluster))
        for line in range(len(cluster)):
            all_clus.append(cluster[line])
    clus_arr = np.array(all_clus)
    print(30*'∂')
    print('Plot %s, cluster per foloder %s'%(plots+1,clus_per_folder))
    print(30*'∂')
# % 
    same_method = 'One space (4D or 5D)' # WE are looking for cluster using 4 or 5 dimesional spaces. If a two different cluster,found in 4 and 5 dimensional space respectively, have at least one commons star, the variable with chage to 'Two method' 
    if len(set(equal_method)) ==2:
        same_method = 'Two spaces (4D and 5D)'
    
    cu_test, index_u = np.unique(clus_arr[:,0:2],return_index=True,axis=0)
    cluster_unique0 = clus_arr[index_u]
   
    
    
    #Here we are going to make a sigma clipping for the cluster members
    #Not really sure which dimension I should clipp...
    sigma =10
    # trimmed_by ='color'#TODO
    # trimmed_by = 'pm'#TODO
    trimmed_by = 'color_pm'#TODO
    if trimmed_by =='color':   
        clus_trim = sigma_clip(cluster_unique0[:,7]-cluster_unique0[:,8],sigma = sigma,maxiters = 10)
        good_filt = np.where(np.isnan(clus_trim)==False)
        cluster_unique = cluster_unique0[good_filt]
    # elif trimmed_by =='pm':
    #     # velocidad = np.sqrt(cluster_unique0[:,4]**2 + cluster_unique0[:,5]**2 )
    #     # clus_trim = sigma_clip(cluster_unique0[:,4],sigma = sigma,maxiters = 10)
    #     clus_trim = sigma_clip(velocidad,sigma = sigma,maxiters = 10)
    #     good_filt = np.where(np.isnan(clus_trim)==False)
    #     cluster_unique = cluster_unique0[good_filt]
    elif trimmed_by =='pm':
        clus_trim_x = sigma_clip(cluster_unique0[:,4],sigma = sigma, maxiters = 10)
        clus_trim_y =  sigma_clip(cluster_unique0[:,5],sigma = sigma, maxiters = 10)
        good_filt = np.where(np.isnan(clus_trim_x)==False &(np.isnan(clus_trim_y)==False))
        cluster_unique = cluster_unique0[good_filt]
    elif trimmed_by =='color_pm':
        clus_trim_color = sigma_clip(cluster_unique0[:,7]-cluster_unique0[:,8],sigma = sigma,maxiters = 10)
        clus_trim_x = sigma_clip(cluster_unique0[:,4],sigma = sigma, maxiters = 10)
        clus_trim_y =  sigma_clip(cluster_unique0[:,5],sigma = sigma, maxiters = 10)
        good_filt = np.where((np.isnan(clus_trim_color)==False) & 
                             np.isnan(clus_trim_x)==False 
                             & (np.isnan(clus_trim_y)==False))
        cluster_unique = cluster_unique0[good_filt]
    # ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks_mean, dAks_mean, radio("),cluster_ID
    cluster_candidate = cluster_unique[:,[0,1,7,8]]
    eso_coor = SkyCoord(ra = [np.mean(cluster_candidate[:,0]/15)], dec = [np.mean(cluster_candidate[:,1])], unit ='deg')
    ra_cen, dec_cen = eso_coor.ra.value, eso_coor.dec.value
    with open(pruebas +'cluster_centroid.csv','w') as file:
           ra_h, dec_g = math.floor(ra_cen), math.ceil(dec_cen)
           ra_min, dec_min = 60 * (ra_cen % 1), 60 * (dec_cen*-1 % 1)
           ra_sec, dec_sec = 60 * (ra_min % 1), 60 * (dec_min % 1)
           file.write('\n'.join(('name,ra,dec,mag',
                                 'Star_%s,%d:%02d:%06.3f,%d:%02d:%06.3f,%s \n' %('Centroid',ra_h, ra_min, ra_sec,dec_g, dec_min, dec_sec,'-'))))
           # file.write('name,ra,dec, mag \n')
           file.close()
    with open(regions +'cluster_centroid.reg','w') as file:
           ra_h, dec_g = math.floor(ra_cen), math.ceil(dec_cen)
           ra_min, dec_min = 60 * (ra_cen % 1), 60 * (dec_cen*-1 % 1)
           ra_sec, dec_sec = 60 * (ra_min % 1), 60 * (dec_min % 1)
           file.write('\n'.join(('# Region file format: DS9 version 4.1',
                                 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                                 'fk5',
                                 'circle(%d:%02d:%06.3f,%d:%02d:%06.3f,5.00") # fill=1'%(ra_h, ra_min, ra_sec,dec_g, dec_min, dec_sec))))
           # file.write('name,ra,dec, mag \n')
           file.close()
    with open(pruebas +'cluster_candidate.csv','w') as file:
        file.write('name,ra,dec,mag'+'\n')
        # file.write('name,ra,dec, mag \n')
        file.close()
        eso_coor.dec.to(u.hourangle).value
    for star_n in range(len(cluster_candidate)):
        ra_h, dec_g = math.floor(cluster_candidate[star_n][0]/15), math.ceil(cluster_candidate[star_n][1])
        ra_min, dec_min = 60 * (cluster_candidate[star_n][0]/15 % 1), 60 * (cluster_candidate[star_n][1]*-1 % 1)
        ra_sec, dec_sec = 60 * (ra_min % 1), 60 * (dec_min % 1)
        mag = cluster_candidate[star_n][3]
        with open(pruebas +'cluster_candidate.csv','a') as file:
            # file.write('%d:%02d:%06.3f \n' % (ra_h,ra_min,ra_sec))
            file.write('Star_%s,%d:%02d:%06.3f,%d:%02d:%06.3f,%s \n' %(star_n+1,ra_h, ra_min, ra_sec,dec_g, dec_min, dec_sec,mag))
            # file.write('Star_%s,%d:%02d:%06.3f,%d:%02d:%06.3f,,,,,,,,,"H=%s", \n' % (star_n+1,ra_h, ra_min, ra_sec,dec_g, dec_min, dec_sec,mag))
    # np.savetxt(pruebas + 'cluster_candidate.txt',cluster_candidate,fmt = 2*'%.8f '+ 2*'%.4f ', header = 'ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks_mean, dAks_mean, radio("),cluster_ID')
    
    # colores_trim = []
    # sig = 2
    # for tr in range(len(gns_core_match)):
    #     colores_trim.append(float(gns_core_match[tr,6]-gns_core_match[tr,8]))
    # print(np.mean(colores_trim))
    # fil_color = sigma_clip(colores_trim, sigma=sig, maxiters=10)
    # good_filt = np.where(np.isnan(fil_color)==False)

    # gns_core_match_trim = gns_core_match[good_filt]
    new_stars = 'no'
    if any(x<len(cluster_unique) for x in cluster_len):
        new_stars = 'yes'
        print('some stars added')
    else:
        print('all cluster are the same')
    plots += 1
    if plots == 1:
        coord_to_save = SkyCoord(ra = cluster_unique[:,0], dec = cluster_unique[:,1],unit = 'deg', frame = 'icrs', equinox = 'J2000', obstime = 'J2014.2')
        np.savetxt(pruebas + 'fine_coord_combined_clus%s_sect%s.txt'%(plots, section),np.array([coord_to_save.ra.value,coord_to_save.dec.value]).T)
        np.savetxt(pruebas + 'combined_clus%s_sect%s.txt'%(plots, section),cluster_unique,fmt ='%.7f',header = 'ra, dec, l, b, pml, pmb,J, H, Ks,x, y, Aks_mean,dAks_mean, radio("), cluster_ID')
        
    np.savetxt(morralla+'combined_Sec_%s_clus/cluster_num/'%(section) + 'cluster_combined_clus%s_sect%s.txt'%(plots, section),cluster_unique)    
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    # fig.suptitle('Sec: %s. In folder --> %s'%(section,os.path.basename(folder)),fontsize=40)
    # ax[0].set_title('Plot #%s. Found %s times. Combiend cluster: %s'%(plots,clus_per_folder,new_stars))
    ax[0].scatter(catal[:,-6],catal[:,-5],color = 'k', alpha = 0.1, zorder=1)
    ax[0].scatter(cluster_unique[:,4],cluster_unique[:,5], color = color_de_cluster, zorder=3,s=100) 
    ax[0].set_xlim(-15,5)
    ax[0].set_ylim(-10,10)
    ax[0].invert_xaxis()
    
    mul_sig, mub_sig = np.std(cluster_unique[:,4]), np.std(cluster_unique[:,5])
    mul_mean, mub_mean = np.mean(cluster_unique[:,4]), np.mean(cluster_unique[:,5])
    
    mul_sig_all, mub_sig_all = np.std(catal[:,-6]), np.std(catal[:,-5])
    mul_mean_all, mub_mean_all = np.mean(catal[:,-6]), np.mean(catal[:,-5])
    
    
    vel_txt = '\n'.join(('$\mu_{l}$ = %s, $\mu_{b}$ = %s'%(round(mul_mean,3), round(mub_mean,3)),
                         '$\sigma_{\mu_{l}}$ = %s, $\sigma_{\mu_{b}}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
    vel_txt_all = '\n'.join(('$\mu_{l}$ = %s, $\mu_{b}$ = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
    
    
    
    prop_0 = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.3)
    prop_01 = dict(boxstyle='round', facecolor='k', alpha=0.5)
    
    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
        verticalalignment='top', bbox=prop_0)
    # ax[0].text(0.05, 0.27, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
    #     verticalalignment='top', bbox=prop_01)
    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =fs) 
    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =fs) 
    
    c2 = SkyCoord(ra = cluster_unique[:,0]*u.deg,dec = cluster_unique[:,1]*u.deg, frame ='icrs', equinox ='J2000', obstime = 'J2014.2')
    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
    rad = max(sep)/2
    
    m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)], frame ='icrs', equinox ='J2000', obstime = 'J2014.2')
    
    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*1.5)
    
    big_idxc, big_group_md, big_d2d,big_d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*5.5)
    
    ax[0].scatter(catal[:,-6][group_md],catal[:,-5][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)
    
    prop_02 = dict(boxstyle='round', facecolor='red', alpha=0.3)
    vel_txt_around = '\n'.join(('$\mu_{l}$ = %s, $\mu_{b}$ = %s'%(round(np.mean(catal[:,-6][group_md]),3), round(np.mean(catal[:,-5][group_md]),3)),
                         '$\sigma_{\mu_{l}}$ = %s, $\sigma_{\mu_{b}}$ = %s'%(round(np.std(catal[:,-6][group_md]),3), round(np.std(catal[:,-5][group_md]),3))))
    ax[0].text(0.05, 0.163, vel_txt_around, transform=ax[0].transAxes, fontsize=30,
        verticalalignment='top', bbox=prop_02)

   

    # ax[1].set_title('max dmu = %s. %s'%(lim, same_method)) 
    prop_1 = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
    ax[1].text(0.15, 0.95, 'aprox. group radio = %s"\nco-moving stars = %s '%(round(rad.to(u.arcsec).value,2),len(cluster_unique)), transform=ax[1].transAxes, fontsize=30,
                            verticalalignment='top', bbox=prop_1)
    
    # ax[1].scatter(catal[:,7][group_md],catal[:,8][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.3)

    # ax[1].scatter(catal[:,7],catal[:,8],color ='k',alpha = 0.1)
    # ax[1].scatter(cluster_unique[:,9], cluster_unique[:,10], color = color_de_cluster,s=100)
    # ax[1].set_xlabel('x',fontsize =30) 
    # ax[1].set_ylabel('y',fontsize =30) 
    
    

    ax[1].scatter(catal[:,5][group_md],catal[:,6][group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.3)
    ax[1].scatter(catal[:,5][big_group_md],catal[:,6][big_group_md], color ='k',alpha = 0.1)

    # ax[1].scatter(ra_,dec_,color ='k',alpha = 0.01)
    ax[1].scatter(cluster_unique[:,0], cluster_unique[:,1], color = color_de_cluster,s=100)
    ax[1].quiver(cluster_unique[:,0], cluster_unique[:,1], cluster_unique[:,4], cluster_unique[:,5], alpha=1, color='k')
    # if section == 'B':
    #     ax[1].scatter(ban_coord.ra, ban_coord.dec,color ='b',s =80,marker ='x')
    #     for ban_star in range(len(ban_coord)):
    #         sep_ban = c2.separation(ban_coord[ban_star])
    #         if min(sep_ban.value) <1/3600:
    #             ax[1].set_facecolor('lavender')
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(4))

    ax[1].set_xlabel('Ra(°)',fontsize =fs) 
    ax[1].set_ylabel('Dec(°)',fontsize =fs,labelpad = -50) 

    
    # Here we are going to match the cluster with gns extion for get the average extiontion and plot an isochrone
    H_Ks_yes = []
    Ks_yes = []
    AKs_clus_all =[]
    clus_coord = SkyCoord(ra = cluster_unique[:,0], dec = cluster_unique[:,1], unit = 'deg', frame ='icrs', equinox ='J2000', obstime = 'J2014.2')
    idx = clus_coord.match_to_catalog_sky(gns_coord)
    validas = np.where(idx[1]<0.5*u.arcsec)
    gns_match = AKs_center[idx[0][validas]]
    for member in range(len(gns_match)):
        if gns_match[member,16] != '-' and gns_match[member,18] != '-':
            AKs_clus_all.append(float(gns_match[member,18]))
          
    AKs_clus, std_AKs = np.median(AKs_clus_all),np.std(AKs_clus_all)
    absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
    AKs = min(AKs_list, key=absolute_difference_function)
    
    iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
    dist = 8200 # distance in parsec
    metallicity = 0.30 # Metallicity in [M/H]
    # # logAge_600 = np.log10(0.61*10**9.)
    logAge = np.log10(0.0050*10**9.)
    logAge1 = np.log10(0.055*10**9.)
    # logAge_30 = np.log10(0.030*10**9.)
    # logAge_60 = np.log10(0.060*10**9.)
    # logAge_90 = np.log10(0.090*10**9.)
    evo_model = evolution.MISTv1() 
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawNoguerasLara18()
    filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
    
    # AKs = 2.0
    iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                        iso_dir=iso_dir)
    # iso1 =  synthetic.IsochronePhot(logAge1, AKs, dist, metallicity=metallicity,
    #                                 evo_model=evo_model, atm_func=atm_func,
    #                                 red_law=red_law, filters=filt_list,
    #                                     iso_dir=iso_dir)
    imf_multi = multiplicity.MultiplicityUnresolved()
    
    H_round = round(gns['Hmag'],4)
    K_round = round(gns['Ksmag'],4)
    mag_uncert = np.zeros((len(cluster_unique),2))
    for i in range(len(cluster_unique)):
        e_mag = np.where((H_round ==  cluster_unique[i][7]) & (K_round == cluster_unique[i][8]) )
        try:
            mag_uncert[i][0] = gns['e_Hmag'][e_mag[0]]
            mag_uncert[i][1] = gns['e_Ksmag'][e_mag[0]]
        except: 
            mag_uncert[i][0] =  mag_uncert[i-1][0]
            mag_uncert[i][1] =  mag_uncert[i-1][1]
        
    error_bars = np.sqrt(mag_uncert[:,0]**2 + mag_uncert[:,1]**2)
    
    # # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01
    
    # # NOTE: when defining the power law slope for each segment of the IMF, we define
    # # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
    # # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 
    
    massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
    powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
    # my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
    my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)
    
    ####################################################
    # ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
    #                   iso.points['m_hawki_Ks'], 'b-',  label='%.1f Myr'%(10**logAge/10**6))
    # ax[2].plot(iso1.points['m_hawki_H'] - iso1.points['m_hawki_Ks'], 
    #                   iso1.points['m_hawki_Ks'], 'royalblue',  label='25 Myr')
    # ax[2].legend(loc =2)
    
    txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
    # ax[2].text(0.65, 0.50, txt_AKs, transform=ax[2].transAxes, fontsize=20,
    #     verticalalignment='top', bbox=prop_01)
    # ax[2].set_title('Trimmed by: %s. %s$\sigma$ (%s stars trimmied)'%(trimmed_by,sigma, len(cluster_unique0)-len(cluster_unique)))
    ax[2].scatter(catal[:,3]-catal[:,4],catal[:,4], color='k' ,s=50,zorder=1, alpha=0.03)
    ax[2].scatter(catal[:,3][group_md]-catal[:,4][group_md],catal[:,4][group_md], color='r' ,s=50,zorder=1, alpha=0.5,marker='x')
    # ax[2].errorbar(cluster_unique[:,7]-cluster_unique[:,8],cluster_unique[:,8],xerr=error_bars,fmt='o')

    ax[2].scatter(cluster_unique[:,7]-cluster_unique[:,8],cluster_unique[:,8], color=color_de_cluster ,s=120,zorder=3, alpha=1)

    ax[2].invert_yaxis()  
    
    ax[2].set_xlabel('$H-Ks$',fontsize =fs)
    ax[2].set_ylabel('$Ks$',fontsize =fs)
    ax[2].set_xlim(1.3,2.5)
    ax[2].set_ylim(max(catal[:,4]),min(catal[:,4]))
    # ax[2].set_title('Cluster %s, eps = %s'%(clus_num,round(eps_av,3)))
    txt_clus = '\n'.join(('H-Ks =%.3f'%(np.mean(cluster_unique[:,7]-cluster_unique[:,8])),
                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(cluster_unique[:,7]-cluster_unique[:,8])),
                         ))
    props_arou = dict(boxstyle='round', facecolor=color_de_cluster, alpha=0.3)
    ax[2].text(0.55, 0.95,txt_clus, transform=ax[2].transAxes, fontsize=30,
        verticalalignment='top', bbox=prop_0)
    
    txt_around= '\n'.join(('H-Ks =%.3f'%(np.mean(catal[:,3][group_md]-catal[:,4][group_md])),
                         '$\sigma_{H-Ks}$ = %.3f'%(np.std(catal[:,3][group_md]-catal[:,4][group_md])),
                         ))
    props_arou = dict(boxstyle='round', facecolor='r', alpha=0.3)
    ax[2].text(0.55, 0.15,txt_around, transform=ax[2].transAxes, fontsize=30,
        verticalalignment='top', bbox=props_arou)
    fig.savefig('/Users/amartinez/Desktop/PhD/ERIS/proposal/cluster.png',dpi=150,format='png',bbox_inches='tight')

    # sys.exit('line 337')

#%%
H_round = round(gns['Hmag'],4)
K_round = round(gns['Ksmag'],4)
mag_uncert = np.zeros((len(cluster_unique),2))
for i in range(len(cluster_unique)):
    e_mag = np.where((H_round ==  cluster_unique[i][7]) & (K_round == cluster_unique[i][8]) )
    try:
        mag_uncert[i][0] = gns['e_Hmag'][e_mag[0]]
        mag_uncert[i][1] = gns['e_Ksmag'][e_mag[0]]
    except: 
        mag_uncert[i][0] =  mag_uncert[i-1][0]
        mag_uncert[i][1] =  mag_uncert[i-1][1]
    print(gns['e_Hmag'][e_mag[0]])

# %%
# mag_uncert[i][1] = gns['e_Hmag'][e_mag[0]]
print(mag_uncert[:,0])
      

