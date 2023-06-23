#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:25:17 2023

@author: amartinez
"""

# The idea is to check the variations around the massive stars where we have
# find a cluster to justify the search areas around the massive stars.

# Si tienes un mapa de extinción del centro galáctico, puedes utilizarlo para estimar las variaciones locales en la extinción alrededor de una fuente específica. Aquí tienes un enfoque general para hacerlo:

# 1. Obtén el valor de extinción del mapa en la ubicación de la fuente que te 
# interesa. Esto te dará una estimación inicial de la extinción en ese punto.

# 2. Examina la resolución espacial del mapa de extinción. Si el mapa tiene una 
# resolución fina, es posible que no necesites hacer ajustes adicionales. 
# Sin embargo, si la resolución es relativamente baja, es posible que desees 
# considerar la interpolación o suavizado para obtener una estimación más
#  precisa de la extinción en el entorno de la fuente.

# 3. Identifica regiones cercanas a la fuente en las que la extinción sea más 
# homogénea. Puedes hacer esto seleccionando un área alrededor de la fuente 
# y examinando los valores de extinción en esa área. Busca regiones donde 
# la extinción sea similar en un rango determinado.

# 4. Calcula un promedio ponderado de los valores de extinción en las regiones 
# seleccionadas. Puedes asignar pesos a cada valor de acuerdo con su proximidad 
# a la fuente o basado en la confiabilidad de los datos en esas regiones. 
# Esto te dará una estimación más precisa de la extinción local alrededor 
# de la fuente.

# 5. Si deseas una estimación aún más precisa, considera utilizar técnicas de 
# modelado que tomen en cuenta la distribución de polvo y gas en la región del 
# centro galáctico. Estos modelos pueden incorporar información adicional, como 
# la estructura tridimensional de la región y las propiedades del medio 
# interestelar, para calcular la extinción local con mayor precisión.

# Recuerda que la estimación de las variaciones locales de la extinción puede 
# estar sujeta a incertidumbres y limitaciones inherentes a los datos y a 
# los modelos utilizados. Por lo tanto, es importante ser cauteloso y considerar 
# todas las fuentes posibles de error en tu análisis.
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
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from extinction_simulation import extinction_now
from scipy.spatial import distance
import matplotlib.colors as colors
from scipy.interpolate import griddata
from matplotlib import cm
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# MS stars with cluster so far
# 14996	154855	954199	9192 	18538
st_list = [14996	,154855,	954199,	9192,18538]
# st_list = [954199]
maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
layer = 2
AKs = fits.open(maps + 'K%sHK_C.fit'%(layer),memmap=True)        
Ks_map = WCS(maps + 'K%sHK_C.fit'%(layer))
# pix_sc = AKs[0].header['CD2_2']
# pix_sc = AKs[0].header['CD2_1']
pix_sc = 20#pix/arcsec
sep =  150*pix_sc# in arcsec. Search area for extinction variation to look around
around = sep*2
with open(pruebas + 'st_map.reg','w') as fil:
    fil.write('# Region file format: DS9 version 4.1\nglobal color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nphysical\n')
for st in st_list:
    mas_ind = np.where(yso[:,2] == st)
    
    mas = yso[mas_ind]
    
    # cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
    
    
    
    cor = [mas[0][0:2][0],mas[0][0:2][1]]
    pix_Ks =   Ks_map.wcs_world2pix([cor],1)
    
    # Pixel scale roughlt correspont to 3 arsec every 60 pixels
    
    x_ar = np.arange(pix_Ks[0][1].astype(int) -around, pix_Ks[0][1].astype(int) + around+1,55)
    y_ar = np.arange(pix_Ks[0][0].astype(int) -around, pix_Ks[0][0].astype(int) + around+1,55)
    xv, yv = np.meshgrid(x_ar,y_ar)
    # %
    

    pos = np.vstack([yv.ravel(), xv.ravel()])
    dist=distance.cdist(np.array(pix_Ks),pos.T, 'euclidean')
    radio = np.where(dist < sep)
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(pos[0],pos[1], color = 'k',s =1)
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]], color = 'r',s =5)
    # ax.set_aspect('equal', 'box')
    # %
    AKs_clus = [AKs[0].data[pos[1][radio[1]][i]][pos[0][radio[1]][i]]
                for i in range(len(radio[1]))]
   
    print('st = %s\nMax diff = %.2f, std = %.2f, radio = %s'%(st, np.nanmax(AKs_clus) - np.nanmin(AKs_clus),np.nanstd(AKs_clus),sep/pix_sc))
    np_plot = np.array([pos[0][radio[1]], pos[1][radio[1]],AKs_clus]).T
# =============================================================================
#     with open(pruebas + 'st_map.reg','a') as fil:
#         fil.write('point(%s,%s) # point = x\n# text(%s,%s) text={%.3f}\ncircle(%s,%s,%s)\n'%(pix_Ks[0][0],pix_Ks[0][1],
#                                                                          pix_Ks[0][0],pix_Ks[0][1]-10,
#                                                                 AKs[0].data[pix_Ks[0][1].astype(int)][pix_Ks[0][0].astype(int)],
#                                                                          pix_Ks[0][0],pix_Ks[0][1],sep))
# 
# =============================================================================
    
# %%   
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    # ax = fig.add_subplot(projection='3d')
    ax[0].set_title('%s'%(st))
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]],c = np.log(np.array(AKs_clus)))
    AKs_c = np.array(AKs_clus)
    ax[0].scatter(pos[0][radio[1]], pos[1][radio[1]],c=AKs_clus,norm=colors.LogNorm(vmin=np.nanmin(AKs_c), vmax=np.nanmax(AKs_c)))

    ax[0].set_aspect('equal', 'box')
    
    np_plot = np.array([pos[0][radio[1]], pos[1][radio[1]],AKs_clus]).T
    
    # %
    
    
    # x = np_plot[:, 0]  # X coordinates (first column)
    # y = np_plot[:, 1]  # Y coordinates (second column)
    # z = np_plot[:, 2]  # Z values (third column)
    
    # buenos = np.where(np.isnan(z) == False)
    # x = x[buenos]
    # y = y[buenos]
    # z = z[buenos]
    
    
    
    # # Create a grid for contour plotting
    # x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    
    # # Perform interpolation to obtain Z values on the grid
    # z_grid = griddata((x, y), z, (x_grid, y_grid), method='nearest')
    
    # # Create a contour plot
    
    # v_min, v_max = 2,4
    # lev =np.arange(v_min,v_max+1,1)
    # ax[1].tricontour(x, y, z, levels=lev, linewidths=2,colors = 'white',linestyles='dashed',vmin=v_min, vmax=v_max)  # Contour lines
    # axcb = ax[1].tricontourf(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), levels=lev,vmin=v_min, vmax=v_max, cmap='viridis')  # Filled contours
    # ax[1].set_aspect('equal', 'box')
    # fig.colorbar(axcb,ax =ax[1],shrink=0.7)
    
# %%

    # x_ = np_plot[:,0]
    # y_ = np_plot[:,1]
    # z_ = np_plot[:,2]
    
    x_ = np_plot[:,0]
    y_ = np_plot[:,1]
    z_ = np_plot[:,2]
    
    buenos = np.where(np.isnan(z_) == False)
    x_ = x_[buenos]
    y_ = y_[buenos]
    z_ = z_[buenos]
    
    center_x = pix_Ks[0][0]
    center_y = pix_Ks[0][1]
    distances = np.sqrt((x_ - center_x) ** 2 + (y_ - center_y) ** 2)
    
    
    
    std_devs = []
    
    unique_distances, ind_t = np.unique(np.round(distances,0),return_index=True)
    # unique_distances = distances
    for dist in unique_distances:
        indices = np.where(distances < dist)
        std_dev = np.nanstd(z_[indices])
        std_devs.append(std_dev)
    std_devs =np.array(std_devs)
    # good_xy = np.where(np.round()) 
    
    d_line  = np.where(std_devs>np.nanmean(std_devs))[0][0]
    unique_distances, ind_t = np.unique(np.round(distances,0),return_index=True)
    fig, ax = plt.subplots(1,1)
    ax.set_ylim(0,0.35)
    ax.set_title(st)
    ax.axhline(np.mean(std_devs),color = 'red')
    ax.axvline(unique_distances[d_line],linestyle='dashed',color ='red')
    ax.text(unique_distances[d_line], 0.10,'%s'%(unique_distances[d_line]) )
    ax.scatter(unique_distances,std_devs)
    
    # fig, ax = plt.subplots(1,1)
    # # ax.set_ylim(0,0.35)
    # ax.set_title(st)
    
    # h =ax.hist2d(unique_distances,std_devs)
    # fig.colorbar(h[3])
    # unique_distances = np.linspace(distances.min(), distances.max(), 100)
    # X, Y = np.meshgrid(unique_distances, unique_distances)
    # Z = griddata((x_[ind_t], y_[ind_t]), std_devs, (X, Y), method='nearest')
    # scaled_Z = Z * 1000
    
    # # Z = np.array(std_devs).reshape(len(unique_distances), len(unique_distances))
    
    
    # # %
    # fig, ax = plt.subplots(1,1)
    # # ax.contour(X,Y,Z,levels=2)
    # # ax.contourf(X, Y, scaled_Z, cmap=cm.jet)
    # plt.figure()
    # plt.contourf(X, Y, scaled_Z, cmap=cm.jet)
    # plt.colorbar()

# %%

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from scipy.interpolate import griddata

# # Create example data (replace with your actual numpy array)
# np.random.seed(0)
# # your_array = np.random.rand(900, 3)
# your_array = np_plot
# # Extract coordinates and Z values
# x = your_array[:, 0]
# y = your_array[:, 1]
# z = your_array[:, 2]

# buenos = np.where(np.isnan(z) == False)
# x = x[buenos]
# y = y[buenos]
# z = z[buenos]

# # Calculate distances from the center point
# center_x = np.mean(x)
# center_y = np.mean(y)
# distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

# # Create a grid for distances
# unique_distances = np.linspace(distances.min(), distances.max(), 100)
# X, Y = np.meshgrid(unique_distances, unique_distances)

# # Interpolate Z values on the grid
# Z = griddata((x, y), z, (X, Y), method='nearest')

# # Scale the Z array for better visualization
# # scaled_Z = Z * 1000

# # Create contour plot with colormap
# plt.figure()
# plt.contourf(X, Y,Z,levels =3)
# plt.xlabel('Distance from Center (X)')
# plt.ylabel('Distance from Center (Y)')
# plt.title('Contour Map of Standard Deviation Variance')
# plt.colorbar()
# plt.show()







