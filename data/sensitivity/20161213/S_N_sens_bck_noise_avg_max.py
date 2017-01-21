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

N = 55

def fit(amps,m,x):
    return m*amps**x
    
SNs = np.array([1.2,1.2,1.25,1.1,0.95,0.65,0.4574,0.2656,0.1467,0.0401,0.0559])
amps = np.array([10.2, 5.1,2.5, 1,0.5,0.25,0.2,0.143,0.1,0.071,0.05])
#SN_err = np.array([ 0.1092, 0.16653, 0.12926,   0.1174, 0.11999, 0.12726,0.06053, 0.04318,0.0417, 0.04097])
SN_err = sqrt((SNs/sqrt(2))**2+1)*np.array([1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),1/sqrt(200),0.05476,0.02983,0.03281,0.01761,0.04757])
amps_plt = np.delete(amps[5:],-2)
SNs_plt = np.delete(SNs[5:],-2)
popt,perr=pt.plot_fit(amps_plt,SNs_plt,fit,fitguess,fit_label='fit',hold=fithold,fmt_data='o',fmt_fit='--',show=False)

plt.errorbar(amps,SNs,yerr=SN_err,linestyle = '', marker='o')

amps_th = np.linspace(0.3,0.01)
plt.subplot()
plt.subplot().set_xscale('log')
plt.subplot().set_yscale('log')

plt.plot(amps_th,fit(amps_th,popt[0],2),'--',label=r'Fit: {:.3f}*z^({:g})'.format(popt[0],2))

dk = (2*pi/(.9*1e-6))
tau = 0.020
ACSS_l = 3.3e4
ACSS_u = 3.3e4
U_u = 2*pi*ACSS_u*2*(np.cos(67.76*pi/180))**2
U_l = 2*pi*ACSS_l*2*(np.cos(65.92*pi/180))**2
U = .5*(U_u+U_l)
DWF = 0.86
G_tot = 72.41
A = np.exp(-tau*G_tot)

m = (DWF*U*dk*tau*1e-9)**2*sqrt(N)/(4*sqrt(2)*sqrt(A**(-2)-1))
plt.plot(amps_th,fit(amps_th,m,2),'--',label=r'Theory with only proj noise (incl bck): {:.3f}*z^({:g})'.format(m,2))

amps_th = np.linspace(0.1,10,1000)
pwr = np.linspace(0.0001,1,1000)

S_N_th = []
for amp in amps_th:
    theta_maxs = pwr*DWF*U*dk*amp*1e-9*tau
    A = np.exp(-pwr*tau*G_tot)
    S_N_m = []
    for p,theta_max in enumerate(theta_maxs):
        if theta_max>1.5:
            theta_max = 1.5
        Pup = 1/2 - 1/2*A[p]*j0(theta_max) 
        sig_proj = 1/sqrt(N)*sqrt(Pup*(1-Pup))
        sig_bck = 1/sqrt(N)*sqrt(0.25*(1-A[p]**2))
        sig_up = (A[p]**2)/8*(1+j0(2*theta_max)-2*j0(theta_max)**2)
        delta_J_m = 2*A[p]**(-1)*sqrt(sig_proj**2 + sig_up + sig_bck**2)
        delta_th_m = delta_J_m/(j1(theta_max))
        S_N_m += [theta_max/delta_th_m/2]
    S_N_max = max(S_N_m)
    S_N_th += [S_N_max]

plt.plot(amps_th,S_N_th,'--',label='Full theory')
plt.legend(loc=(1,0))
plt.ylim(0,2)
#plt.xlim(0,.5)
plt.xlabel('Displacement Amplitude (nm)')
plt.ylabel('Signal to noise')