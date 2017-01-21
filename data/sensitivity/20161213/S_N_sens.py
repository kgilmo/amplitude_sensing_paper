# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:26:58 2016

@author: kag3
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_tools as pt
from numpy import pi, sqrt, log
from scipy.special import j0,j1

fitguess = np.array([11,2])
fithold = np.array([False,True])

def fit(amps,m,x):
    return m*amps**x
    
SNs = np.array([1.2,1.2,1.25,1.1,1.05,0.8,0.4973,0.3281,0.1897,0.0552,0.078])
amps = np.array([10.2, 5.1,2.5, 1,0.5,0.25,0.2,0.143,0.1,0.071,0.05])
#SN_err = np.array([ 0.1092, 0.16653, 0.12926,   0.1174, 0.11999, 0.12726,0.06053, 0.04318,0.0417, 0.04097])
SN_err = sqrt((SNs/sqrt(2))**2+1)*np.array([1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(3000),1/sqrt(3000),1/sqrt(3000),1/sqrt(3000),1/sqrt(3000)])
amps_plt = np.delete(amps[5:],-2)
SNs_plt = np.delete(SNs[5:],-2)
popt,perr=pt.plot_fit(amps_plt,SNs_plt,fit,fitguess,fit_label='fit',hold=fithold,fmt_data='o',fmt_fit='--',show=False)

plt.errorbar(amps,SNs,yerr=SN_err,linestyle = '', marker='o')
s
amps_th = np.linspace(0.5,0)
plt.subplot()
plt.subplot().set_xscale('log')
plt.subplot().set_yscale('log')

plt.plot(amps_th,fit(amps_th,popt[0],2),'--',label=r'{:.3f}*z^({:g})'.format(popt[0],2))
plt.plot(amps_th,fit(amps_th,27.34,2),'--',label=r'{:.3f}*z^({:.3f})'.format(27.34,2))



dk = (2*pi/(.9*1e-6))
tau = 0.024
ACSS_l = 3.3e4
ACSS_u = 3.3e4
U_u = 2*pi*ACSS_u*2*(np.cos(67.76*pi/180))**2
U_l = 2*pi*ACSS_l*2*(np.cos(65.92*pi/180))**2
U = .5*(U_u+U_l)
DWF = 0.86
amps_th = np.linspace(0.2,0.01,100)
theta_max = DWF*U*dk*amps_th*1e-9*tau
G_tot = 67.613
A = np.exp(-tau*G_tot)

theta_max_2 = np.linspace(2,2,300)
amps_th_2 = np.linspace(10,0.25,300)
theta_max = np.append(theta_max_2,theta_max)
amps_th = np.append(amps_th_2,amps_th)
Pup = 1/2 - 1/2*A*j0(theta_max) 
sig_proj = 1/sqrt(55)*sqrt(Pup*(1-Pup))
sig_up = (A**2)/8*(1+j0(2*theta_max)-2*j0(theta_max)**2)
delta_J_m = 2*A**(-1)*sqrt(sig_proj**2 + sig_up)
delta_th_m = delta_J_m/(j1(theta_max))
S_N_m = theta_max/delta_th_m/2

plt.plot(amps_th,S_N_m,'--')

plt.legend(loc=(1,0))
plt.ylim(0,2)
#plt.xlim(0,.5)
plt.xlabel('Displacement Amplitude (nm)')
plt.ylabel('Signal to noise')