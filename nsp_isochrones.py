#!/usr/bin/env python
# coding: utf-8

# In[2]:


from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import numpy as np
import pylab as py
import pdb
import matplotlib.pyplot as plt

# =============================================================================
# ISOCHRONE CONSTRUCTOR
# =============================================================================
# In[5]:


# AKs = [1.6, 1.65, 1.7,1.74,1.75,1.8,1.85,1.9,1.95,2.0,2.05,2.10] # extinction in mags
AKs = np.arange(1.6,2.11,0.01)
# AKs = [0]
dist = 8200 # distance in parsec
metallicity = [0.30,0.17] # Metallicity in [M/H]

# Define evolution/atmosphere models and extinction law
evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()

# Also specify filters for synthetic photometry (optional). Here we use 
# the HST WFC3-IR F127M, F139M, and F153M filters
filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']

# Specify the directory we want the output isochrone
# table saved in. If the directory does not already exist,
# SPISEA will create it.
iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
# logAge = [np.log10(0.010*10**9.), np.log10(0.030*10**9.),np.log10(0.060*10**9.),np.log10(0.090*10**9.), np.log10(0.5*10**9.),np.log10(0.61*10**9.),np.log10(0.80*10**9.),np.log10(1*10**9.)]
logAge = [np.log10(0.0025*10**9.),np.log10(0.0048*10**9.),np.log10(0.0015*10**9.),np.log10(0.0055*10**9.),np.log10(0.0020*10**9.)]

for a in AKs:
    for b in logAge:
        for c in metallicity:
            iso_nsd = synthetic.IsochronePhot(b, a, dist, metallicity=c,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                        iso_dir=iso_dir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




