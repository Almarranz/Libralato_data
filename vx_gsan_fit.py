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
import pandas as pd
import sys
import matplotlib.patches as patches
import matplotlib.patches as mpatches
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
rcParams.update({'font.size': 30})
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

section = 'All'
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
df_np=np.loadtxt(resultados + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# df_np=np.loadtxt(resultados + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))

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
lim_dmul=2
accu=np.where((abs(dmul)<lim_dmul) & (abs(dmub)<lim_dmul))#Are they in the paper selecting by the error of the galactic or equatorial coordintes???

#%
mul=mul[accu]
mub=mub[accu]
dmul=dmul[accu]
dmub=dmub[accu]

#%
print(min(mul),max(mul))
binwidth=0.25
auto='no'
if auto !='auto':
    auto=np.arange(min(mul),max(mul)+ binwidth, binwidth)#also works if running each bing width one by one, for some reason...
    # print(auto)


fig, ax = plt.subplots(1,1, figsize=(10,10))

# sig_h=sigma_clip(mul,sigma=500,maxiters=20,cenfunc='mean',masked=True)
# mul=mul[sig_h.mask==False]

h=ax.hist(mul,bins=auto,linewidth=2,density=True)
h1=np.histogram(mul,bins=auto,density=False)

x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
ax.axvline(np.mean(mul), color='r', linestyle='dashed', linewidth=3)
ax.legend(['List=%s, %s, mean= %.2f, std=%.2f'
                  %(name,len(mul),np.mean(mul),np.std(mul))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
ax.set_ylabel('N')
ax.set_xlim(-13,3)
ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
ax.invert_xaxis()
y=h[0]#height for each bin

#%


# This plots mub vs mul  
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
#%

yerr=[]
y=np.where(y==0,0.001,y)
y1=h1[0]
y1=np.where(y1==0,0.001,y1)
# yerr = y*np.sqrt(1/y1+1/len(mul))
yerr = y*np.sqrt(1/y1)
# 

#%   
def gaussian(x, mu, sig, amp):
    return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) 
#%
def loglike(theta):
    mu1, sigma1, amp1,mu2,sigma2,amp2,mu3,sigma3,amp3 = theta
    model = gaussian(x, mu1, sigma1, amp1)+gaussian(x,mu2,sigma2,amp2)+gaussian(x,mu3,sigma3,amp3)
     
    return -0.5 * np.sum(((y - model)/yerr) ** 2)#chi squared model

#% 

def prior_transform(utheta):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    #x = 2. * u - 1.  # scale and shift to [-1., 1.)
    #x *= 10.  # scale to [-10., 10.)
    umu1, usigma1, uamp1,  umu2, usigma2, uamp2, umu3, usigma3, uamp3= utheta
     
#%     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
    mu1 = -4*umu1-4/2  # yellow
    sigma1 = 3* (usigma1)   
    amp1 = 1 * uamp1 
   
    mu2 = -5*umu2-5/2 # red
    sigma2 =4*usigma2   #for section 'All' sigma2 is better 4, for section 'A' works better with 3-3.5
    amp2 = .5* uamp2   

    mu3 = -7*umu3-7/2 # black
    sigma3 = 3*(usigma3)
    amp3 = 1*uamp3
        
    return mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3
    
#%
sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=9, nlive=200,
                                            bound='multi', sample='rwalk')
sampler.run_nested()
res = sampler.results

#%
from dynesty import plotting as dyplot

labels = [r'$\mathrm{\mu 1}$', r'$\mathrm{\sigma 1}$', r'$amp1$', r'$\mathrm{\mu 2}$', r'$\mathrm{\sigma 2}$', r'$amp2$', 
              r'$\mathrm{\mu 3}$', r'$\mathrm{\sigma 3}$', r'$amp3$']
    # fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
    #                              fig=plt.subplots(6, 2, figsize=(16, 27)))
    
fig, axes = dyplot.traceplot(sampler.results,labels=labels,show_titles=True,
                                 fig=plt.subplots(9, 2, figsize=(16, 20)))
    
plt.show()



#%%
from dynesty import utils as dyfunc
    
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
print(mean)
# %
sp=np.ones(9)*0.68
fig, axes = dyplot.cornerplot(res, color='royalblue' ,show_titles=True,quantiles=[0.16,0.5,0.84],truths=mean,
                                  quantiles_2d=[0.16,0.5,0.84],
                                  title_kwargs=({'x': 0.65, 'y': 1.05}), labels=labels,
                                  fig=plt.subplots(9, 9, figsize=(28, 28)))

plt.legend(['$\mu_{l}$ %s'%(name)],fontsize=70,markerscale=0,shadow=True,bbox_to_anchor=(1.2,9.8),handlelength=-0.0)
plt.show() 
#% 

results = sampler.results
print(results['logz'][-1])
# %%
# h=plt.hist(v_x*-1, bins= nbins, color='darkblue', alpha = 0.6, density =True, histtype = 'stepfilled')
h=plt.hist(mul, bins= auto, color='slategrey', alpha = 0.5, density =True, histtype = 'stepfilled')

xplot = np.linspace(min(x), max(x), 1000)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.hist(mul, bins= auto, color='slategrey', alpha = 0.5, density =True, histtype = 'stepfilled')
ax.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5])
         + gaussian(xplot, mean[6], mean[7], mean[8]), color="darkorange", linewidth=3, alpha=1)
ax.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="blue", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot, gaussian(xplot, mean[6], mean[7], mean[8]) , color='black', linestyle='dashed', linewidth=3, alpha=0.6)

# ax.set_ylim(0,0.16)
ax.set_xlim(-19,10)

# %%
# =============================================================================
# # plt.text(-10,max(h[0]-0.01),'logz=%.0f'%(results['logz'][-1]),color='b')
# =============================================================================
qu = [dyfunc.quantile(samps, [0.16,0.5,.84], weights=weights)
             for samps in samples.T]
ax.invert_xaxis()
#ths box
prop_0 = dict(boxstyle='round', facecolor='white' , alpha=0.2)
bulge = ('$\mu_{l}$=%.2f$\pm$%.2f, $\sigma_{\mu_{l}}$=%.2f$\pm$ %.2f'%(mean[3],qu[3][0]-qu[3][2],mean[4],qu[4][0]-qu[4][2]))
east =  ('$\mu_{l}$=%.2f$\pm$%.2f, $\sigma_{\mu_{l}}$=%.2f$\pm$ %.2f'%(mean[0],qu[0][0]-qu[0][2],mean[1],qu[1][0]-qu[1][2]))
west =  ('$\mu_{l}$=%.2f$\pm$%.2f, $\sigma_{\mu_{l}}$=%.2f$\pm$ %.2f'%(mean[6],qu[6][0]-qu[6][2],mean[7],qu[7][0]-qu[7][2]))

# ax.text(0.02, 0.95, bulge, transform=ax.transAxes, fontsize=15,
#     verticalalignment='top',color='red')
# ax.text(0.02, 0.90, east, transform=ax.transAxes, fontsize=15,
#     verticalalignment='top',color='blue')
# ax.text(0.02, 0.85, west, transform=ax.transAxes, fontsize=15,
#     verticalalignment='top',color='k')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel('N', fontsize = 30)
ax.set_xlabel(r'$\mathrm{\mu_{l}\,(mas\ yr^{-1})}$',fontsize = 30) 
article = '/Users/amartinez/Desktop/PhD/My_papers/Libralato/'
plt.savefig(article + 'nsd_limdmu_l%s.png'%(lim_dmul), dpi=300,bbox_inches='tight')
with open(article +'nsd_values_limdmu%s.txt'%(lim_dmul), 'w') as arch:
    arch.write('#1.Absolute 2. Relative 3. Error\n#Bulge: pm , dispersion. East:pm , dispersion. West: pm , dispersion \n%.2f %.2f %.2f %.2f %.2f %.2f\n%.2f %.2f %.2f %.2f %.2f %.2f '%(mean[3],mean[4],mean[0],mean[1],mean[6],mean[7],mean[3]-mean[3],mean[4],mean[0]-mean[3],mean[1],mean[6]-mean[3],mean[7]))
with open(article +'nsd_values_limdmu%s.txt'%(lim_dmul), 'a') as arch:
    arch.write('\n%.2f %.2f %.2f %.2f %.2f %.2f'%(qu[3][0]-qu[3][2],qu[4][0]-qu[4][2],qu[0][0]-qu[0][2],qu[1][0]-qu[1][2],qu[6][0]-qu[6][2],qu[7][0]-qu[7][2]))
# sys.exit('281')
#%%
print(plt.xticks()[0])
# %%
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
# print(mean)
quantiles = [dyfunc.quantile(samps, [0.16,0.5,.84], weights=weights)
             for samps in samples.T]
# for i in range(9):
#     print('medin %.2f -+ %.2f %.2f'%(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1]))
#     print(' mean %.2f -+ %.2f %.2f'%(mean[i],mean[i]-quantiles[i][0],quantiles[i][2]-mean[i])+'\n'+30*'*')

print(quantiles[3][0]-quantiles[3][2])

# %%

fun1= lambda x: (mean[2] * (1 / (mean[1] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[0], 2.) / (2 * np.power(mean[1], 2.))) )
    # result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-15,15)
gau1=integrate.quad(fun1,-20,10)

fun2= lambda x: (mean[5] * (1 / (mean[4] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[3], 2.) / (2 * np.power(mean[4], 2.))) )
# result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-20,10)
gau2=integrate.quad(fun2,-20,10)

fun3= lambda x: (mean[8] * (1 / (mean[7] * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mean[6], 2.) / (2 * np.power(mean[7], 2.))) )
# result = integrate.quad(gaussian(x, mean[0], mean[1], mean[2]),-20,10)
gau3=integrate.quad(fun3,-20,10)

print(len('Area under Gaus1: %.3f')*'&')
print('Area under Gaus1:%.3f'%(gau1[0]))
print('Area under Gaus2:%.3f'%(gau2[0]))
print('Area under Gaus3:%.3f'%(gau3[0]))
print('Total area = %.3f'%(gau1[0]+gau2[0]+gau3[0]))
print(len('Area under Gaus1: %.3f')*'&')
# np.savetxt(pruebas + 'gaus_mul_sec_%s.txt'%(section),np.array([[mean[0],mean[3],mean[6],mean[1],mean[4],mean[7],gau1[0],gau2[0],gau3[0]]]),fmt='%.3f',header ='mul_e, mul_b, mu_w, sig_e, sig_b, sig_w, area_e, area_b,area_w')
# =============================================================================
# bs = np.arange(min(mul), max(mul), 0.25)
# hist, edges = np.histogram(mul, bs)
# freq = hist/float(hist.sum())
# plt.bar(bs[:-1], freq, width=0.25)
# plt.xlim(-20,10)
# 
# =============================================================================
# =============================================================================
# l = [3,3,3,2,1,4,4,5,5,5,5,5]
# print(len(l))
# bins=np.arange(0,7,1)
# print(bins)
# h_test=plt.hist(l,density=True)
# plt.show()
# =============================================================================
# %%

fig, ax = plt.subplots(1,1,figsize =(10,5))
h=ax.hist(mul, bins= auto, color='royalblue', alpha = 0.6, density =True, histtype = 'stepfilled')

xplot = np.linspace(min(x), max(x), 1000)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

ax.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5])
         + gaussian(xplot, mean[6], mean[7], mean[8]), color="darkorange", linewidth=3, alpha=1)
ax.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="yellow", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot, gaussian(xplot, mean[6], mean[7], mean[8]) , color='black', linestyle='dashed', linewidth=3, alpha=0.6)
propiedades = dict(boxstyle='round', facecolor='r', alpha=0.5)
texto ='\n'.join(('%s $\mathrm{(mas\ yr^{-1})}$'%(round(mean[3],2)),'            $\downarrow$','      %s Km/s'%(round(mean[3]*40)))) 
ax.text(0.75, 0.45, texto, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades)



plt.xlim(-20,10)
# =============================================================================
# # plt.text(-10,max(h[0]-0.01),'logz=%.0f'%(results['logz'][-1]),color='b')
# =============================================================================

plt.gca().invert_xaxis()

plt.ylabel('N')
plt.xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'nsd_dyn_bulpm.png', dpi=300,bbox_inches='tight')

# %%


fig, ax = plt.subplots(1,1,figsize =(10,5))
h=ax.hist(mul-mean[3], bins= auto, color='royalblue', alpha = 0.6, density =True, histtype = 'stepfilled')

xplot = np.linspace(min(x), max(x), 1000)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

ax.plot(xplot-mean[3], gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5])
         + gaussian(xplot, mean[6], mean[7], mean[8]), color="darkorange", linewidth=3, alpha=1)
ax.plot(xplot-mean[3], gaussian(xplot, mean[0], mean[1], mean[2])  , color="yellow", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot-mean[3], gaussian(xplot, mean[3], mean[4], mean[5])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
ax.plot(xplot-mean[3], gaussian(xplot, mean[6], mean[7], mean[8]) , color='black', linestyle='dashed', linewidth=3, alpha=0.6)
propiedades_y = dict(boxstyle='round', facecolor='y', alpha=0.5)
texto_y ='\n'.join(('%s $\mathrm{mas\ yr^{-1}}$'%(round(mean[0]-mean[3],2)),'            $\downarrow$','      %s Km/s'%(round((mean[0]-mean[3])*40)))) 
ax.text(0.05, 0.85, texto_y, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades_y)
propiedades_b = dict(boxstyle='round', facecolor='k', alpha=0.2)
texto_b ='\n'.join(('%s $\mathrm{mas\ yr^{-1}}$'%(round(mean[6]-mean[3],2)),'            $\downarrow$','      %s Km/s'%(round((mean[6]-mean[3])*40)))) 
ax.text(0.45, 0.45, texto_b, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades_b)


plt.xlim(-20,10)
# =============================================================================
# # plt.text(-10,max(h[0]-0.01),'logz=%.0f'%(results['logz'][-1]),color='b')
# =============================================================================

plt.gca().invert_xaxis()

plt.ylabel('N')
plt.xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 

# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'nsd_dyn_nsdpm.png', dpi=300,bbox_inches='tight')






























