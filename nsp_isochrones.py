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


AKs = [1.6, 1.65, 1.7,1.75,1.8,1.85,1.9,1.95,2.0,2.05,2.10] # extinction in mags
dist = 8000 # distance in parsec
metallicity = 0.30 # Metallicity in [M/H]

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

for a in AKs:
    logAge = np.log10(0.61*10**9.) # Age in log(years)
    iso_nsd = synthetic.IsochronePhot(logAge, a, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




