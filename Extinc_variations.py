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
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

yso=np.loadtxt(cata+'GALCEN_TABLE_D.cat',usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')

# MS stars with cluster so far
# 14996	154855	954199	9192 	18538
# st_list = [14996	,154855,	954199,	9192,18538]
st_list = [14996]
maps = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
layer = 2
AKs = fits.open(maps + 'K%sHK_C.fit'%(layer),memmap=True)        
Ks_map = WCS(maps + 'K%sHK_C.fit'%(layer))
# pix_sc = AKs[0].header['CD2_2']
# pix_sc = AKs[0].header['CD2_1']
pix_sc = 20#pix/arcsec
sep =  20*pix_sc# in arcsec. Search area for extinction variation to look around
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
    radio = np.where(dist < sep*10)
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(pos[0],pos[1], color = 'k',s =1)
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]], color = 'r',s =5)
    # ax.set_aspect('equal', 'box')
    # %
    AKs_clus = [AKs[0].data[pos[1][radio[1]][i]][pos[0][radio[1]][i]]
                for i in range(len(radio[1]))]
    
    print('st = %s\nMax diff = %.2f, std = %.2f, radio = %s'%(st, np.nanmax(AKs_clus) - np.nanmin(AKs_clus),np.nanstd(AKs_clus),sep/pix_sc))

# =============================================================================
#     with open(pruebas + 'st_map.reg','a') as fil:
#         fil.write('point(%s,%s) # point = x\n# text(%s,%s) text={%.3f}\ncircle(%s,%s,%s)\n'%(pix_Ks[0][0],pix_Ks[0][1],
#                                                                          pix_Ks[0][0],pix_Ks[0][1]-10,
#                                                                 AKs[0].data[pix_Ks[0][1].astype(int)][pix_Ks[0][0].astype(int)],
#                                                                          pix_Ks[0][0],pix_Ks[0][1],sep))
# 
# =============================================================================
    # x_grid, y_grid = np.meshgrid(pos[0][radio[1]], pos[1][radio[1]])
    # z_grid = np.reshape(z, (len(y), len(x)))
    fig, ax = plt.subplots(1,1)
    # ax = fig.add_subplot(projection='3d')
    ax.set_title('%s'%(st))
    # ax.scatter(pos[0][radio[1]], pos[1][radio[1]],c = np.log(np.array(AKs_clus)))
    AKs_c = np.array(AKs_clus)
    ax.scatter(pos[0][radio[1]], pos[1][radio[1]],c=AKs_clus,norm=colors.LogNorm(vmin=AKs_c.min(), vmax=AKs_c.max()))
    ax.set_aspect('equal', 'box')
    # ax.contour(pos[0][radio[1]], pos[1][radio[1]],np.array(AKs_clus))
    # sys.exit('137')
    np_plot = np.array([pos[0][radio[1]], pos[1][radio[1]],AKs_clus]).T
    
    # %%
    from scipy.interpolate import griddata
    # Assuming your numpy array is called 'data' with shape (900, 3)
    x = np_plot[:, 0]  # X coordinates (first column)
    y = np_plot[:, 1]  # Y coordinates (second column)
    z = np_plot[:, 2]  # Z values (third column)
    
    # Create a grid for contour plotting
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    
    # Perform interpolation to obtain Z values on the grid
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='linear')
    
    # Create a contour plot
    lev = 2
    plt.tricontour(x, y, z, levels=lev, linewidths=1, colors='white')  # Contour lines
    plt.tricontourf(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), levels=lev, cmap='viridis')  # Filled contours
    plt.colorbar()  # Add a colorba
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot')
    plt.show()
# %%

# %%
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


X_ar, Y_ar = np.meshgrid(x_ar, y_ar)
Z_ar = AKs_clus

x2_ar = np.linspace(min(x_ar), max(x_ar), 30)
y2_ar = np.linspace(min(y_ar), max(y_ar), 30)
f = interp2d(x_ar, y_ar, Z_ar, kind='linear')
Z2_ar = f(x2_ar, y2_ar)

fig, ax = plt.subplots(nrows=1, ncols=1)
# ax[0].pcolormesh(X_ar, Y_ar, Z_ar)

X2_ar, Y2_ar = np.meshgrid(x2_ar, y2_ar)
ax.pcolormesh(X2_ar, Y2_ar, Z2_ar)
# ax.contourf(X_ar, Y_ar, Z2_ar,colors='white')
ax.contour(X_ar, Y_ar, Z2_ar,colors='white', alpha=1)

ax.set_aspect('equal', 'box')
# %%
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 13)
y = np.array([0, 2, 3, 3.5, 3.75, 3.875, 3.9375, 4])
X, Y = np.meshgrid(x, y)
Z = np.sin(np.pi*X/2) * np.exp(Y/2)

x2 = np.linspace(0, 4, 65)
y2 = np.linspace(0, 4, 65)
f = interp2d(x, y, Z, kind='cubic')
Z2 = f(x2, y2)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].pcolormesh(X, Y, Z)

X2, Y2 = np.meshgrid(x2, y2)
ax[1].pcolormesh(X2, Y2, Z2)
ax[1].contourf(X2, Y2, Z2)
plt.show()
# %%
x = np.arange(1, 10)
y = x.reshape(-1, 1)
h = x * y
#