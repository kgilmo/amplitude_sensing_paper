# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:42:25 2015

@author: jgb
"""

import os, importlib
import numpy as np
from numpy import pi, sqrt, sin
import matplotlib.pyplot as plt

import hfGUIdata as hf
import plot_tools as pt
#import squeeze_func as squ
from scipy.special import j0

#options
verbose = False
save = False
raw = True
avg_scans = True
title = r't$_{evolve}$: 20 ms, 0.2 ODF power, vs drive amp'

files_to_use = [1,2,3,4,0]

base_path = os.getcwd()
fns = [os.listdir(base_path)[i] for i in files_to_use]
fns = [i for i in fns if not i.startswith('.DS_Store')]

def fun_cpmg_sat(f,A):
    return 1/2 - 1/2*A*np.exp(-1/4*sig_max*(np.sinc(T*((drive_detun)))*np.sin(np.pi*f*(T+t+t_s))*np.sin((np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun)))
      *(np.cos(2*(np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun))))*(np.cos(4*(np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun)))) )**2)  
 
def fun_cpmg_j(f,A):
    return 1/2 - 1/2*A*j0(sig_max*(np.sinc(T*((drive_detun)))*np.sin(np.pi*f*(T+t+t_s))*np.sin((np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun)))
      *(np.cos(2*(np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun))))*(np.cos(4*(np.pi*f*(T+t+t_s)+np.pi*T*(drive_detun)))) )   )  
     
def fun_cpmg_j2(A):
    return 1/2 - 1/2*A*j0(sig_max*(np.sinc(T*((drive_detun)))*np.cos(np.pi*T*(drive_detun))
      *(np.cos(2*(np.pi*T*(drive_detun))))*(np.cos(4*(np.pi*T*(drive_detun)))) )   )       
      
raman_df = []
Pup = []
err = []
drive_freq = []
avg_ODF_pwr_frac = []
names = []
As = []
sig_max_all = []
z_all = []

#theory calc info
t_s = 5.6*1e-6
dk = (2*pi/(.9*1e-6))

# loop through different data sets here
for i,fn in enumerate(fns):
    folder = os.path.join(base_path,fn)
    os.chdir(folder)
    files = os.listdir(os.getcwd())
    print( "______________Data set: {}__________________".format(hf.n_slice(fn)))
    
    bm = hf.get_ionProp_value('detection%det_brightMean')
    dm = hf.get_ionProp_value('detection%det_darkMean')
    G_tot = 0.5*( hf.get_ionProp_value('raman%align%spontem%G_du') +
                    hf.get_ionProp_value('raman%align%spontem%G_ud') +
                    hf.get_ionProp_value('raman%align%spontem%G_el') )
    k = bm-dm
    det_t = hf.get_ionProp_value('detection%det_t')
    N = k/(hf.get_ionProp_value('detection%det_phtperionperms')*det_t*1e-3)
    J1kHz = hf.get_ionProp_value('raman%raman_J1kHz')
    t = hf.get_ionProp_value('sf%fitParams%sf_fitParam_tpi')*1e-6
    l_pwr = hf.get_ionProp_value('raman%align%lower_odf_voltage')/hf.get_ionProp_value('raman%align%lower_odf_voltage_max')
    u_pwr = hf.get_ionProp_value('raman%align%upper_odf_voltage')/hf.get_ionProp_value('raman%align%upper_odf_voltage_max')
    u_ang = 180-2*(hf.get_ionProp_value('raman%wp%raman_wp_upper_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_lower_home_offset'))
    l_ang = 2*(hf.get_ionProp_value('raman%wp%raman_wp_lower_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_upper_home_offset'))
    mVpp = hf.get_ionProp_value('raman%force_sensing%drive_amp_mVpp')
    if i is 0:
        mVpp = 0
    if i is 4:
        mVpp = 10.1
    z = (mVpp*1e-9)/2    
    ACSS_l = l_pwr*3.21e4
    ACSS_u = u_pwr*3.1e4
    ACSS_err = 500
    ang_err = 0
    U0 = 2*pi*1/2*(ACSS_l + ACSS_u)*2*(np.cos(65*pi/180))**2
    U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
    U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
    U = .5*(U_u+U_l)
    
    name, scandata, counts_data, data = hf.get_raw_counts_hist(combine_scans=avg_scans)
    os.chdir(base_path)
    
    T = data['arm_t'][0]*1e-6
    Jbar = J1kHz*0.001/T
    bp = ((counts_data-dm)/k)
    tau = T*2*8 # !!!!!! Look at sequence name to determine sequence length
    reps = np.shape(bp)[1]
    bp_err = np.std(bp, axis=1)/sqrt(reps)
    bp_avg = np.mean(bp,axis=1)
    if i is 0:
        mag_bck = bp_avg
    #theory params
    DWF = 0.86
    sig_max = DWF*U*dk*z*tau
    gamma = l_pwr*(G_tot)
    A = np.exp(-tau*(gamma))
    
    #add all the data 
    Pup.append(bp_avg)
    err.append(bp_err)
    raman_df.append(scandata*1e3)
    drive_freq.append((data['f'][0]-30)*1e6)
    avg_ODF_pwr_frac.append((l_pwr+u_pwr)/2.)
    As.append(A)
    sig_max_all.append(sig_max)
    z_all.append(z)
 #Load data messages

    print( "Ion number from loading calibration: {:.0f}".format(N))
    print( "Gtot: {:4g}, Jbar: {:4g}, Det_t: {:.3g} ms, tau_total: {:.3g} ms".format(G_tot, 
                                                          Jbar, (det_t*1e-3), tau*1e3))

    names.append(fn[5:-7])   
    
#%% Data analysis
for i,data in enumerate(Pup):

    l = '{:.1f}'.format(z_all[i]*1e9)
    shape = ['v','d','^','s','o']    
    plt.errorbar(raman_df[i]-drive_freq[i],data,yerr=err[i],marker=shape[i],linestyle='',label=l)
    if i is not 0:
        sig_max = sig_max_all[i]
        drive_detun =  np.linspace((raman_df[i]-drive_freq[i])[0],(raman_df[i]-drive_freq[i])[-1],1000)
#        print(raman_df[i]-drive_freq[i])
        plt.plot(drive_detun,fun_cpmg_j2(As[i]),'--',color='#0b1924')

plt.legend(fontsize=9,loc=(.83,.56),title='$Z_{c}$ (nm)')
plt.ylim(0)
#plt.xlim(100,200)
#plt.title(title)
plt.ylabel(r'Bright Population $<P_{\uparrow}>$')
plt.xlabel(r'ODF Detuning $(\mu-\omega)/(2\pi)$ (Hz)')
plt.xlim(-300,300)
plt.ylim(0.0,0.7)
plt.grid(False)
#plt.savefig(r'C:\Users\kag3\Documents\GitHub\amplitude_sensing_paper\figures\lineshape_hires.png',dpi=300,transparent=True)
