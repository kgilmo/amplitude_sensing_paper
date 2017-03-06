# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:23:59 2016

@author: kag3
"""

import os, shutil, os.path
import numpy as np
from numpy import pi, sin, cos, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy.special import j0,j1
#from scicons import pi, hbar, m_Be, k_b,e

import hfGUIdata as hf
import plot_tools as pt
#import squeeze_func as squ

avg_scans = True

tau_fac = 16

files_to_use = [0,3]  
raw = True
save = False
name = "ODF Scan"
# make a copy of the analysis at the folder
if save is True:
    shutil.copy(__file__, os.path.normpath(os.getcwd()))
    
#theory calc info
dk = (2*pi/(.9*1e-6))
    
def fun_J(theta_max):
    return 1/2 - 1/2*A*j0(theta_max)  
    
def root(x,y):
    return j0(x)-y

def root_1(w):
    return -j1(w)

a = 0.0
b = 4.49

fitguess = np.array([G_tot,0.02])
fithold = np.array([False,False])
def bck_fit(odf_pwr,G_tot,mag_bck):
    return 1/2 - 1/2*np.exp(-tau*G_tot*odf_pwr)*np.exp(-0.5*mag_bck)
        
base_path = os.getcwd()
add_path = ""
fns = [os.listdir(os.path.join(base_path,add_path))[i] for i in files_to_use]
num_files = len(fns)

# containers for data sets
N      = np.zeros(num_files)
J1ks   = np.zeros(num_files)
k   = np.zeros(num_files)

odf_pwr  = [0]*(num_files)
counts    = [0]*(num_files)
names  = [0]*(num_files)
hist   = [0]*(num_files)
err   = [0]*(num_files)

#_____________________________________________________________________
# data processing here
for i,fn in enumerate(fns):
    folder = os.path.join(base_path,add_path,fn)
    print(folder)
    os.chdir(folder)
    files = os.listdir(os.getcwd())
    
 #Load properties data
    prop_name = [x for x in files if "_props.csv" in x][0]
    file_name, data_p = hf.get_gen_csv(prop_name, skip_header=False)
    bm = data_p['det_brightMean']
    dm = data_p["det_darkMean"]
    det_t = data_p["det_t"]
    J1ks[i] = data_p['raman_J1kHz'] 
    fz = data_p[('raman_fz')]*1e3
    Ncal = data_p['det_phtperionperms']
    k[i] = bm-dm  # phtns per N atoms
    N[i] = k[i]/(det_t*1e-3)/Ncal
    t = data_p['sf_fitParam_tpi']*1e-6
    u_ang = 180-2*(hf.get_ionProp_value('raman%wp%raman_wp_upper_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_lower_home_offset'))
    l_ang = 2*(hf.get_ionProp_value('raman%wp%raman_wp_lower_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_upper_home_offset'))
    G_tot = 0.5*( hf.get_ionProp_value('raman%align%spontem%G_du') +
        hf.get_ionProp_value('raman%align%spontem%G_ud') +
        hf.get_ionProp_value('raman%align%spontem%G_el') )
    mVpp = hf.get_ionProp_value('raman%force_sensing%drive_amp_mVpp')
    z = (mVpp*0.97e-9)/2

#load experiment specific properties
    for file in os.listdir(os.getcwd()):
        if file.startswith("histData.") and file.endswith(".csv"):
            file_name = file
    props = hf.parse_raw_counts_data(file_name)[1]
    T = props['arm_t']*1e-6
    drive_freq = (props['f']-30)*1e6
    tau = np.mean(props['arm_t'])*1e-6*tau_fac
    mu  = props['raman_df']*1e3
    
#mean counts and odf powers
    data_name = [x for x in files if "_data.csv" in x][0]
    file_name, data = hf.get_gen_csv(data_name, skip_header=True)
    odf_pwr[i] = np.array(data.T[0][0:],dtype='float')
#    counts[i] = np.array(data.T[1][0:],dtype='float')
#    trials = np.array(data.T[3][0:],dtype='float')
#    err[i] = np.array(data.T[2][0:],dtype='float')/np.sqrt(trials)

    # Load raw data
    data_name = [x for x in files if "_raw.csv" in x][0]
    hdata = np.genfromtxt(data_name, delimiter=",", dtype='float')
    reps = np.shape(hdata)[1]
    pts = np.shape(hdata)[0]
    hdata_re = np.reshape(hdata,(pts,1,reps))
    bp_raw_avg = np.ones(int(pts))
    bp_raw_err = np.ones(int(pts))
    bp_raw_std = np.ones(int(pts))
    bp_raw_full = np.ones((int(pts),int(reps-1)))
    
    for j,row in enumerate(hdata_re):
        counts_data = hf.parse_raw_counts(row)
        bp_raw = ((counts_data-dm)/k[i])
        bp_raw_avg[j] = np.mean(bp_raw)
        bp_raw_err[j] = np.std(bp_raw)/sqrt(np.size(bp_raw))
        bp_raw_std[j] = np.std(bp_raw)
        bp_raw_full[j] = bp_raw
    if i is 0:
        label = 'Background ' + fn[:-7]
        
        bck_raw = bp_raw
        bck = bp_raw_avg
        bck_err = bp_raw_std
        bp_raw_bck = bp_raw_full
        popt,perr=pt.plot_fit(odf_pwr[0],bck,bck_fit,fitguess,yerr=bp_raw_err,data_label=label,fit_label='fit',show_fit_param=False,show=True,hold=fithold,fmt_data='d',fmt_fit='--')
#        plt.errorbar(odf_pwr[0],bck,yerr=bp_raw_err,linestyle = '', marker='d',label=label,color='#A60628')     

    if i is 1:
        z_l = z*2e9 
        label = '{:g} mVpp signal at {:g} kHz'.format(z_l,drive_freq*1e-3)   
        
    if i is 2:
        z_l = z*2e9 
        label = '{:g} mVpp signal at {:g} kHz'.format(z_l,drive_freq*1e-3)      
        
    if i is 3:
        z_l = z*2e9 
        label = '{:g} mVpp signal at {:g} kHz'.format(z_l,drive_freq*1e-3)           
    if i is not 0:
        plt.errorbar(odf_pwr[i],bp_raw_avg,yerr=bp_raw_err,linestyle = '', marker='o',label=label,color='#348ABD')     

    #theory params
        odf_th_pwr = np.linspace(0,1,101)
        ACSS_l = odf_th_pwr*3.4e4
        ACSS_u = odf_th_pwr*3.3e4
        U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
        U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
        U = .5*(U_u+U_l)
        DWF = 0.86
        gamma = odf_th_pwr*popt[0]
        gamma_exp = odf_pwr[1]*popt[0]
        A = np.exp(-tau*gamma)*np.exp(-0.5*popt[1])
        A_exp = np.exp(-tau*gamma_exp)*np.exp(-0.5*popt[1])
        theta_max_th = DWF*U*dk*z*tau
        plt.plot(odf_th_pwr,fun_J(theta_max_th),'--',label='Theory',color='#0b1924')    
#        plt.plot(odf_th_pwr,bck_fit(odf_th_pwr,G_tot,popt[1]),'--',label='Theory bck',color='#0b1924')
        
        subs = (bp_raw_avg- bck)/A_exp
        diffs = 1-2*subs
        theta_max =  np.ones(int(len(bp_raw_avg)))
        for l,diff in enumerate(diffs):
            if l is 28:
                a = 4.49
                b = 7.725
            try:
                theta_max[l] = brentq(root,a,b,args=(diff,))
            except ValueError:
                print('Hmm somethings not quite right... setting theta_max=',0,l)
                theta_max[l] = 0
        theta_max2 = theta_max**2 
        
    os.chdir(base_path)

#plt.title(r'8 pi CPMG, 24 ms total time '+ fn[:-7])
plt.title('')
plt.xlabel(r'ODF Coupling $F_{0}/F_{0M}$')
plt.ylabel(r'Bright Population $<P_{\uparrow}>$')
plt.ylim(0,0.6)
plt.xlim(0,1.1)
plt.grid(False)
#plt.legend(title='N = {:.0f}, wr = 180 kHz'.format(N[i]), loc=(1,0))

ACSS_l = odf_pwr[1]*3.4e4
ACSS_u = odf_pwr[1]*3.3e4
U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
U = .5*(U_u+U_l)
Z_c = theta_max/(DWF*U*tau*dk)*1e9 
Z_c_2 = Z_c**2
fitguess = np.array([1,1])
fithold = np.array([False,False])
def j_fit(pwr,s,t_max):
    return s*j0(pwr*t_max)

th_opt,th_err= pt.plot_fit(odf_pwr[1],diffs,j_fit,fitguess,data_label=label,fit_label='fit',show_fit_param=False,show=False,hold=fithold,fmt_data='d',fmt_fit='--')

#theta_max_err = (2/A_exp)*sqrt(bp_raw_std**2 + bck_err**2)/abs(th_opt[0]*j1(odf_pwr[1]*th_opt[1]))
theta_max_err = (2/A_exp)*np.std(bp_raw_full - bp_raw_bck,axis=1)/(sqrt(reps-1)*abs(th_opt[0]*j1(odf_pwr[1]*th_opt[1])))
#theta_max_err = (2/A_exp)*np.std(bp_raw_full,axis=1)/abs(th_opt[0]*j1(odf_pwr[1]*th_opt[1]))
Z_c_err = theta_max_err/(DWF*U*tau*dk)*1e9
Z_c_err_2 = 2*z*1e9*Z_c_err
S_N = theta_max/theta_max_err/2
#plt.show()
#plt.close()
plt.axes([.57, .2, .31, .31])
plt.errorbar(odf_pwr[1][3:-10],Z_c_2[3:-10],yerr=Z_c_err_2[3:-10],linestyle = '', marker='.')
Z_c_th_2 = (np.full_like(odf_pwr[1],z*1e9))**2
plt.plot(odf_pwr[1],Z_c_th_2,'r--')  
plt.yticks(np.arange(0, 0.51, 0.25))
xlabels = ['0.0','','','','0.8']
plt.xticks(np.arange(0, 0.801, 0.2),xlabels)
#plt.xlabel(r'$U/U_{max}$')
plt.ylabel(r'$Z_{c}^{2} (\rm{nm}^{2})$')
plt.ylim(-0.05,0.5)
plt.xlim(0,0.8)
plt.grid(False)

#plt.savefig(r'C:\Users\kag3\Documents\GitHub\amplitude_sensing_paper\figures\measurement_strength_inset_z2.png',dpi=300,transparent=True)
#plt.plot(odf_pwr[1],S_N,'o')
#
#plt.plot(odf_pwr[1],diffs,'o')
#plt.plot(odf_pwr[1],th_opt[0]*j0(odf_pwr[1]*th_opt[1]))