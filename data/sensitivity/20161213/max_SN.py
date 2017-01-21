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
import squeeze_func as squ
from scipy.optimize import curve_fit, brentq
from scipy.special import j0,j1
#options
verbose = False
save = False
raw = True
avg_scans = False

files_to_use = [0,2,4,6,8,10]
bck_files = [1,3,5,7,9,11]

base_path = os.getcwd()
fns = [os.listdir(base_path)[i] for i in files_to_use]
fns = [i for i in fns if not i.startswith('.DS_Store')]
num_fns = len(fns)

bck_fns = [os.listdir(base_path)[i] for i in bck_files]
bck_fns = [i for i in bck_fns if not i.startswith('.DS_Store')]
num_bck_fns = len(bck_fns)

a = 0
b = 3.8317

odf_pwr = []
bp_sig = []
bp_bck = []
err = []
mVpps = []
names = []
As = []
sig_max_all = []
stds = []

#theory calc info
t_s = 5.6*1e-6
dk = (2*pi/(.9*1e-6))

def root(x,y):
    return j0(x)-y
    
# loop through different data sets here
for i,fn in enumerate(fns):
    folder = os.path.join(base_path,fn)
    os.chdir(folder)
    files = os.listdir(os.getcwd())
    print( "______________Data set: {}__________________".format(fn[:-4]))
    
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
    z = (mVpp*1e-9)/2    
    ACSS_l = l_pwr*3.4e4
    ACSS_u = u_pwr*3.4e4
    U0 = 2*pi*1/2*(ACSS_l + ACSS_u)*2*(np.cos(65*pi/180))**2
    U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
    U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
    U = .5*(U_u+U_l)
    
    name, scandata, counts_data, data = hf.get_raw_counts_hist(combine_scans=avg_scans)
    os.chdir(base_path)
    
    T = data['arm_t']*1e-6
    Jbar = J1kHz*0.001/T
    bp = ((counts_data-dm)/k)
    tau = T*2*8 # !!!!!! Look at sequence name to determine sequence length
    reps = np.shape(bp)[0]
    bp_std = np.std(bp)
    bp_avg = np.mean(bp)
    
    #theory params
    DWF = 0.86
    sig_max = DWF*U*dk*z*tau
    A = np.exp(-tau*G_tot)
    #add all the data 
    bp_sig += [bp]
    odf_pwr.append(scandata)
    mVpps.append(mVpp)
    sig_max_all.append(sig_max)
    As.append(A)
    
 #Load data messages

    print( "Ion number from loading calibration: {:.0f}".format(N))
    print( "Gtot: {:4g}, Jbar: {:4g}, Det_t: {:.3g} ms, tau_total: {:.3g} ms".format(G_tot, 
                                                          Jbar, (det_t*1e-3), tau*1e3))

    names.append(fn[5:-4])  
    
l = "{:.3g} mVpp".format(mVpps[0]) + ' (' + names[0] + ', ' + names[1][7:] + ', ' + names[2][7:] + ', ' + names[3][7:] + ', ' + names[4][7:] + ', ' + names[5][7:] + ')'

bp_sig_avg = np.mean(bp_sig)
bp_sig_err = np.std(bp_sig)/sqrt(2*reps*num_fns)
bp_sig_std = np.std(bp_sig)

# loop through different data sets here
for i,fn in enumerate(bck_fns):
    folder = os.path.join(base_path,fn)
    os.chdir(folder)
    files = os.listdir(os.getcwd())
    print( "______________Data set: {}__________________".format(fn[:-4]))
    
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
    z = (mVpp*1e-9)/2    
    ACSS_l = l_pwr*3.4e4
    ACSS_u = u_pwr*3.4e4
    U0 = 2*pi*1/2*(ACSS_l + ACSS_u)*2*(np.cos(65*pi/180))**2
    U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
    U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
    U = .5*(U_u+U_l)
    
    name, scandata, counts_data, data = hf.get_raw_counts_hist(combine_scans=avg_scans)
    os.chdir(base_path)
    
    T = data['arm_t']*1e-6
    Jbar = J1kHz*0.001/T
    bp = ((counts_data-dm)/k)
    tau = T*2*8 # !!!!!! Look at sequence name to determine sequence length
    reps = np.shape(bp)[0]
    bp_std = np.std(bp)
    bp_avg = np.mean(bp)
    
    #theory params
    DWF = 0.86
    sig_max = DWF*U*dk*z*tau
    A = np.exp(-tau*G_tot)
    #add all the data 
    bp_bck += [bp]
    odf_pwr.append(scandata)
    mVpps.append(mVpp)
    sig_max_all.append(sig_max)
    As.append(A)
 #Load data messages

    print( "Ion number from loading calibration: {:.0f}".format(N))
    print( "Gtot: {:4g}, Jbar: {:4g}, Det_t: {:.3g} ms, tau_total: {:.3g} ms".format(G_tot, 
                                                          Jbar, (det_t*1e-3), tau*1e3))

    names.append(fn[5:-4])  
    
bp_bck_avg = np.mean(bp_bck)
bp_bck_err = np.std(bp_bck)/sqrt(2*reps*num_bck_fns)
bp_bck_std = np.std(bp_bck)

l_bck = "{:.3g} mVpp".format(mVpps[-1]) + ' (' + names[6] + ', ' + names[7][7:] + ', ' + names[8][7:] + ', ' + names[9][7:] + ', ' + names[10][7:] + ', ' + names[11][7:] + ')'

#%% Data analysis
plt.errorbar(odf_pwr[i],bp_bck_avg,yerr=bp_bck_err,linestyle = '', marker='o',label=l_bck)     
plt.errorbar(odf_pwr[i],bp_sig_avg,yerr=bp_sig_err,linestyle = '', marker='o',label=l)     

plt.legend(fontsize=9,loc=(1,0))
plt.title(r'8 pi CPMG, {:.1f} ms total time '.format(tau*1e3))
plt.xlabel('Fractional ODF Power')


sub = (bp_sig_avg - bp_bck_avg)/(1-2*bp_bck_avg)
diff = 1-2*sub
theta_max = brentq(root,a,b,args=(diff,))
theta_max2 = theta_max**2    
delta_F = np.std(bp_sig)/(1-2*bp_bck_avg)
F_deriv = 0.25*j1(sqrt(theta_max2))/sqrt(theta_max2)
delta_theta2 = delta_F/F_deriv
S_N2 = theta_max2/delta_theta2

S_N = (bp_sig_avg - bp_bck_avg)/sqrt(bp_sig_std**2+bp_bck_std**2)

print(S_N)
print(bp_sig_err)

