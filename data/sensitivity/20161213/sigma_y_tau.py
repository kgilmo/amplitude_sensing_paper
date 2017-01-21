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
max_avg = 300
files_to_use = [0,2,4,6,8,10]
#files_to_use = [1,3,5,7,9,11]

base_path = os.getcwd()
fns = [os.listdir(base_path)[i] for i in files_to_use]
fns = [i for i in fns if not i.startswith('.DS_Store')]
num_fns = len(fns)

odf_pwr = []
bp_sig = []
bp_bck = []
err = []
mVpps = []
names = []
As = []
sig_max_all = []
stds = []
    
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
    
    name, scandata, counts_data, data = hf.get_raw_counts_hist(combine_scans=avg_scans)
    os.chdir(base_path)
    
    T = data['arm_t']*1e-6
    Jbar = J1kHz*0.001/T
    bp = ((counts_data-dm)/k)
    tau = T*2*8 # !!!!!! Look at sequence name to determine sequence length
    reps = np.shape(bp)[0]
    bp_std = np.std(bp)
    bp_avg = np.mean(bp)
    
    #add all the data 
    bp_sig += [bp]

    
 #Load data messages

    print( "Ion number from loading calibration: {:.0f}".format(N))
    print( "Gtot: {:4g}, Jbar: {:4g}, Det_t: {:.3g} ms, tau_total: {:.3g} ms".format(G_tot, 
                                                          Jbar, (det_t*1e-3), tau*1e3))

    names.append(fn[5:-4])  
    
avg_nums = np.arange(1,max_avg+1)
numsinavg = np.arange(0,max_avg+1)
trial_devs = []

bp_sig_avg = np.ones(int(num_fns))
bp_sig_std = np.ones(int(num_fns))

for rep, bp_rep in enumerate(bp_sig):
    bp_sig_avg[rep] = np.mean(bp_rep)
    bp_sig_std[rep] = np.std(bp_rep)
    allan_devs = []
    
    for num in avg_nums:
        avg_bps = []
        reps = []
        
        #do some averaging
        for j in range(0,len(bp_rep)-num+1,num):
            add_bps = []
            for k in range(j,j+num):
                add_bps += [bp_rep[k]]
                
            avg = np.mean(add_bps)
            avg_bps += [avg]
            reps += [j]   
            
        #calculate allan deviation and ACSS fluctuations
        allan_dev = np.sqrt(.5*np.mean((np.roll(avg_bps,1)-avg_bps)**2))
    
        allan_devs += [allan_dev]
        
    trial_devs += [allan_devs]    
#%% Data analysis
for i,no in enumerate(bp_sig):
    x = np.linspace(avg_nums[0],avg_nums[-1])
    y = allan_devs[0]*x**(-1/2)
    plt.loglog(avg_nums,trial_devs[i],'o')
    plt.loglog(x,y,'--',label='-1/2 power law')
    plt.title(r'Trial {}, {}'.format(i+1,names[i]))
    plt.xlabel('Number of experiments averaged')
    plt.ylabel('Allan Deviation')
    plt.legend(loc='lower right')
    plt.ylim(10**(-6),)
    plt.show()
    plt.close()