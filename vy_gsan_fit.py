#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:52:00 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
import scipy.integrate as integrate
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter
from io import StringIO
import pandas as pd
import sys
# %%plotiin parametres
from matplotlib import rcParams
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
resultados='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID

# name='ACSWFC'
name='WFC3IR'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
        
section = 'A'
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
df_np=np.loadtxt(resultados + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))# df_np=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))

gal_coor=df_np[:,[17,18,19,20]]#mul,mub,dmul,dmub
#%%

# %%

# df_np=df.to_numpy()

valid=np.where(np.isnan(df_np[:,4])==False)

df_np=df_np[valid]
gal_coor=gal_coor[valid]

# center=np.where(df_np[:,-2]-df_np[:,4]>2.5)#Color cut usied by libralato
center=np.where(df_np[:,3]-df_np[:,4]>1.3)#Color cut used by Paco
df_np=df_np[center]
gal_coor=gal_coor[center]

ra=df_np[:,5]
dec=df_np[:,6]
mua=df_np[:,9]
mud=df_np[:,11]
dmua=df_np[:,10]
dmud=df_np[:,12]

mul=gal_coor[:,0]
mub=gal_coor[:,1]

dmul=gal_coor[:,2]
dmub=gal_coor[:,3]

#%%

#%%

#%

lim_dmul=1
accu=np.where((abs(dmul)<lim_dmul) & (abs(dmub)<lim_dmul))
#%%
mul=mul[accu]
mub=mub[accu]
dmul=dmul[accu]
dmub=dmub[accu]
# m139=m139[accu]
#%%
print(min(mul),max(mul))

auto='no'
if auto !='auto':
    auto=np.arange(min(mul),max(mul),0.25)#also works if running each bing width one by one, for some reason...
    # print(auto)

#%%


#%%
fig, ax = plt.subplots(1,1, figsize=(10,10))

# sig_hb=sigma_clip(mub,sigma=500,maxiters=20,cenfunc='mean',masked=True)
# mub=mub[sig_hb.mask==False]

hb=ax.hist(mub,bins=auto,color='orange',linewidth=2,density=True)
hb1=np.histogram(mub,bins=auto,density=False)

xb=[hb[1][i]+(hb[1][1]-hb[1][0])/2 for i in range(len(hb[0]))]#middle value for each bin
ax.axvline(np.mean(mub), color='r', linestyle='dashed', linewidth=3)
ax.legend(['List=%s, %s, mean= %.2f, std=%.2f'
                  %(name,len(mub),np.mean(mub),np.std(mub))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
ax.set_ylabel('N')
ax.set_xlim(-10,10)
ax.set_xlabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
yb=hb[0]#height for each bin
#%%

# =============================================================================
# fig, ax =plt.subplots(1,1,figsize=(10,10))
# ax.scatter(mul,mub,color='k',s=1,alpha=0.05)
# ax.set_xlim(-13,2)
# ax.set_ylim(-10,10)
# ax.axvline(0)
# ax.axhline(0)
# ax.axhline(-0.22)
# ax.invert_xaxis()
# 
# =============================================================================
# %%

yerr=[]
yb=np.where(yb==0,0.001,yb)
y1=hb1[0]
y1=np.where(y1==0,0.001,y1)
yerr = yb*np.sqrt(1/y1)
# yerr = y*np.sqrt(1/y1+1/len(v_y))

    
# In[7]:   
def gaussian(x, mu, sig, amp):
    return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) 
#%%
def loglike(theta):
    mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
    model = gaussian(xb, mu1, sigma1, amp1)+gaussian(xb,mu2,sigma2,amp2)
     
    return -0.5 * np.sum(((yb - model)/yerr) ** 2)#chi squared model
#%% 
def prior_transform(utheta):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    #x = 2. * u - 1.  # scale and shift to [-1., 1.)
    #x *= 10.  # scale to [-10., 10.)
    umu1, usigma1, uamp1,  umu2, usigma2, uamp2= utheta

#     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
    mu1 = 1* umu1-0.5  # scale and shift to [-10., 10.)
    sigma1 = 5* (usigma1)   
    amp1 = 1 * uamp1 

    
    mu2 = 2 * umu2-1
    sigma2 = 2 * usigma2   
    amp2 = 1* uamp2   
    

    return mu1, sigma1, amp1, mu2, sigma2, amp2
#%% 
sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=6, nlive=500,
                                            bound='multi', sample='rwalk')
sampler.run_nested()
res = sampler.results
#%%
from dynesty import plotting as dyplot
rcParams.update({'font.size': 10})
# truths = [mu1_true, sigma1_true, amp1_true, mu2_true, sigma2_true, amp2_true]
labels = [r'$\mathrm{\mu 1}$', r'$\mathrm{\sigma 1}$', r'$amp1$', r'$\mathrm{\mu 2}$', r'$\mathrm{\sigma 2}$', r'$amp2$']
# fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
#      $\mathrm{\mu_{b}}                        fig=plt.subplots(6, 2, figsize=(16, 27)))

fig, axes = dyplot.traceplot(sampler.results,labels=labels,show_titles=True,
                             fig=plt.subplots(6, 2, figsize=(20, 16)))



plt.show()
rcParams.update({'font.size': 20})

#%%
from dynesty import utils as dyfunc
    
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
print(mean)
     #%%                              fig=plt.subplots(6, 6, figsize=(28, 28)))
    # This is de corner plot
fig, axes = dyplot.cornerplot(res, color='royalblue', show_titles=True, quantiles=[0.16,0.5,0.68],truths=mean,
                              quantiles_2d=[0.16,0.5,0.68],
                              title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
                              fig=plt.subplots(6, 6, figsize=(28, 28)))
plt.legend(['$\mu_{b}$ %s '%(name)],fontsize=70,markerscale=0,shadow=True,bbox_to_anchor=(1,6.5),handlelength=-0.0)

plt.show() 
    
# %%

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})


results = sampler.results
print(results['logz'][-1])

fig, ax = plt.subplots(figsize=(8,8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
h=ax.hist(mub, bins= auto, color='royalblue', alpha = 0.6, density =True, histtype = 'stepfilled')


xplot = np.linspace(min(xb), max(xb), 1000)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=1)
plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="k", linestyle='dashed', linewidth=3, alpha=0.6)

plt.text(5,max(hb[0]-0.05),'$logz=%.0f$'%(results['logz'][-1]),color='b')

plt.xlim(-15,15)

plt.ylabel('N')
plt.xlabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$')
# %%
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
# print(mean)
quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights)
             for samps in samples.T]

for i in range(6):
    print('medin %.2f -+ %.2f %.2f'%(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1]))
    print(' mean %.2f -+ %.2f %.2f'%(mean[i],mean[i]-quantiles[i][0],quantiles[i][2]-mean[i])+'\n'+30*'*')
   
# %%
fun1= lambda x: (mean[2] * (1 / (mean[1] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[0], 2.) / (2 * np.power(mean[1], 2.))) )
# result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
gau1=integrate.quad(fun1,-15,15)

fun2= lambda x: (mean[5] * (1 / (mean[4] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[3], 2.) / (2 * np.power(mean[4], 2.))) )
# result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
gau2=integrate.quad(fun2,-15,15)

# fun3= lambda x: (mean[8] * (1 / (mean[7] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[6], 2.) / (2 * np.power(mean[7], 2.))) )
# # result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
# gau3=integrate.quad(fun3,-15,15)
print(len('Area under Gaus1: %.3f')*'&')
print('Area under Gaus1:%.3f'%(gau1[0]))
print('Area under Gaus2:%.3f'%(gau2[0]))

print('Total area = %.3f'%(gau1[0]+gau2[0]))
print(len('Area under Gaus1: %.3f')*'&')
np.savetxt(pruebas + 'gaus_mul_sec_%s.txt'%(section),np.array([[mean[0],mean[3],mean[1],mean[4]]]),fmt='%.3f',header ='mub_nsd, mub_b, sig_nsd, sig_b,')






