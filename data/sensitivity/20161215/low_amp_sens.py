# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:07:29 2016

@author: jgb
"""
import os, csv
import numpy as np
from numpy import pi, sqrt, sin
import matplotlib.pyplot as plt

import plot_tools as pt
import hfGUIdata as hf
from scipy.optimize import curve_fit, brentq
from scipy.special import j0,j1

def root(x,y):
    return j0(x)-y
    
a = 0
b = 3.8317

SNs = [] 
SN2s = []
errs = []
err2s = []

def parse_raw_counts(array):
    chan = np.copy(array)
    bad = 0
    for x in np.nditer(array, op_flags=['readwrite']):
        chan[...] = int(bin(x)[3:7],2)
        #x[...] = int(x) & 0x1fff
        try: 
            x[...] = int(bin(x)[7:],2)
        except ValueError:
            x[...] = 0
    if bad > 0:
        print("# of bad points: {}".format(bad))

files_to_use = [0,1,2,3,5,6,7,8,9,10,11,12]
fn = hf.get_files_to_use(os.getcwd(), files_to_use)[0]
amps = [0.2, 0.1, 0.025, 0.05, 0.159, 0.08,10,5,2.5,0.25,0.5,1.0]
base_path = os.getcwd()
fns = [os.listdir(base_path)[i] for i in files_to_use]
fns = [i for i in fns if not i.startswith('.DS_Store')]
num_fns = len(fns)
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
    nm = z*1e9    
    ACSS_l = l_pwr*3.4e4
    ACSS_u = u_pwr*3.4e4
    U0 = 2*pi*1/2*(ACSS_l + ACSS_u)*2*(np.cos(65*pi/180))**2
    U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
    U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
    U = .5*(U_u+U_l)
    print("# of Ions: {:.3g}, J1kHz: {:.5g}, amp: {:.3f}".format(N,J1kHz,z*1e9))
    
    #extract # of experiments
    os.chdir('params')
    expt_param_name = False
    for file in os.listdir(os.getcwd()):
        if file.endswith(".txt"):
            expt_param_name = file
    if expt_param_name is False:
        print("Did not find file")
    expt_params = np.genfromtxt(expt_param_name, 
                                dtype='str', 
                                delimiter=' = ',
                                comments=None)
    ind = np.where(expt_params == '{'+"Num Exp"+'}')
    val_str = expt_params[ind[0].astype(int)[0]][1]
    trials = int(np.float(val_str[1:-1]))
    os.chdir('..')
        
    #parse the counts
    file_name = False
    for file in os.listdir(os.getcwd()):
        if file.startswith("rawData.") and file.endswith(".csv"):
            file_name = file
    if file_name is False:
        print("Did not find file")
    
    else:
    
        with open(file_name) as f:
            reader = csv.reader(f)
            colnames = next(reader)
    
        colnames = [i for i in colnames if i] # gets rid of column names that are empty strings
        num_reg_columns = len(colnames) - 1
        alldata = np.genfromtxt(file_name, delimiter=",", names=True, dtype=None, usecols=range(num_reg_columns+1))
        non_hist_cols = num_reg_columns  # defined by HFGUI expt type
        
        counts_data = np.genfromtxt(file_name, delimiter=",", 
                                    skip_header=1, 
                                    dtype='float')
        if len(np.shape(counts_data))>1:
            num_cols_total = len(counts_data[0])
        else:
            num_cols_total = len(counts_data)
        counts_cols = range(num_reg_columns, num_cols_total)
        counts_data = np.genfromtxt(file_name, delimiter=",", 
                                    skip_header=1, 
                                    dtype='int', 
                                    usecols=counts_cols)
    
    #actally parse the counts data
        copy_cd = np.copy(counts_data)
        scanpts, detections = np.shape(counts_data)
        
        if detections == trials:
            print('single PMT channel used')
            PMTchannels = 1
        else:
            PMTchannels = int(detections/trials)
            print(str(PMTchannels),' channels used')
            counts_channels = np.reshape(counts_data,(scanpts, trials, PMTchannels))
            #check that the reshape was correct, first channel should eval to zero
            if np.all(counts_channels[...,0] >> 17 == 0):
                print('reshape check passed')
            else:
                print('Error in reshaping')
            parse_raw_counts(counts_channels)
        
        os.chdir('..')
    
    #display the data
    scan_val = alldata['dummy']
    total_det_t = alldata[0][-2]
    det_t_per_chan = total_det_t/PMTchannels
    data = np.mean(counts_channels, axis=1)
    bp = ((counts_channels-dm)/k)
    bp_avg = np.mean(bp, axis=1)
    err = np.std(bp, axis=1)/sqrt(trials)
    bck_err = np.delete(err,1,axis=1)
    sig_err = np.delete(err,0,axis=1)
    bp_bck = np.array([val for sublist in np.delete(bp,1,axis=2) for val in sublist]).flatten()
    bp_sig = np.array([val for sublist in np.delete(bp,0,axis=2) for val in sublist]).flatten()

    #theory params
    tau = 0.002
    t_s = 5.6*1e-6
    dk = (2*pi/(.9*1e-6))
    DWF = 0.86
    sig_max = DWF*U*dk*z*tau
    
    print("Total detection time ", total_det_t)
    print("det_t_per_chan ", det_t_per_chan)
    
    for j,scanpt in enumerate(bp_avg.T):
        if j is 0:
            bck = scanpt
            l = 'Background'
            yerr = bck_err
        if j is 1:
            sig = scanpt
            l = r'{:.3f} nm displacement'.format(nm)
            yerr = sig_err
#        plt.errorbar(scan_val,scanpt,yerr=yerr,linestyle = '', marker='o',label=l)
#    plt.legend()
#    plt.xlabel('Scan Value')
#    plt.ylabel('Bright Fraction')
#    plt.ylim(0,.6)
#    plt.show()
#    plt.close()
    
    bck_avg = np.mean(bp_bck)
    bck_std = np.std(bp_bck)
    sig_avg = np.mean(bp_sig)
    sig_std = np.std(bp_sig)
    
    theta_max2 = (bp_sig - bp_bck)/(1-2*bck_avg)
    S_N = theta_max2/np.std(theta_max2)
    
    max_avg = 300
    avg_nums = np.arange(1,max_avg+1)
    numsinavg = np.arange(0,max_avg+1)
    
    bp_sig_avg = np.ones(int(num_fns))
    bp_sig_std = np.ones(int(num_fns))
        
    allan_devs = []
    
    for num in avg_nums:
        avg_SN = []
        reps = []
        
        #do some averaging
        for j in range(0,len(S_N)-num+1,num):
            add_SN = []
            for k in range(j,j+num):
                add_SN += [S_N[k]]
                
            avg = np.mean(add_SN)
            avg_SN += [avg]
 
            
        #calculate error
        SN_err = np.std(avg_SN)/np.sqrt(len(avg_SN))
    
    SNs += [np.mean(S_N)]
    
    errs += [SN_err]

#    S_N2 = (bp_sig - bp_bck)/sqrt(sig_std**2+bck_std**2)
#    SN2s += [np.mean(S_N2)]
#    err2s += [np.std(S_N2)/sqrt(len(S_N2))]
stat_errs = sqrt((SNs/sqrt(2))**2+1)/sqrt(3000)
 
plt.errorbar(amps,SNs,yerr=stat_errs,linestyle = '', marker='o',label='S/N with error from stats')
plt.errorbar(amps,SNs,yerr=errs,linestyle = '', marker='o',label='S/N with error from binning 300 pts')

plt.subplot()
plt.subplot().set_xscale('log')
plt.subplot().set_yscale('log')


dk = (2*pi/(.9*1e-6))
tau = 0.020
ACSS_l = 3.5e4
ACSS_u = 3.5e4
U_u = 2*pi*ACSS_u*2*(np.cos(67.76*pi/180))**2
U_l = 2*pi*ACSS_l*2*(np.cos(65.92*pi/180))**2
U = .5*(U_u+U_l)
DWF = 0.86
A = np.exp(-tau*G_tot)

N=100
m = (DWF*U*dk*tau*1e-9)**2*sqrt(N)/(4*sqrt(2)*sqrt(A**(-2)-1))
#plt.plot(amps_th,fit(amps_th,m,2),'--',label=r'Theory with only proj noise (incl bck): {:.3f}*z^({:g})'.format(m,2))

amps_th = np.linspace(0.01,10,100)
pwr = np.linspace(0.0001,1,100)

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
plt.ylim(0.0,2)
plt.xlabel('Displacement Amplitude (nm)')
plt.ylabel('Signal to noise')