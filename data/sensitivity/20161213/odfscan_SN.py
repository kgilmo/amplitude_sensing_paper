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

import hfGUIdata as hf
import plot_tools as pt
import squeeze_func as squ

avg_scans = True

tau_fac = 16

files_to_use = [1,0]
raw = True
save = False
name = "ODF Scan"
# make a copy of the analysis at the folder
if save is True:
    shutil.copy(__file__, os.path.normpath(os.getcwd()))
    
#theory calc info
t_s = 5.6*1e-6
dk = (2*pi/(.9*1e-6))

a = 0
b = 3.8317

    
def root(x,y):
    return j0(x)-y
    
#response functions without small angle approx, good for saturation. Used to get initial fit parameters for classical drive.
def fun_cpmg_sat(f,B_c):
    return 1/2 - 1/2*A*np.exp(-1/2*B_c*(np.sinc(T*((f-mu)))*np.sin(np.pi*f*(T+t+t_s))*np.sin(np.pi*f*(T+t+t_s)+np.pi*T*(f-mu)))**2)   
   
def fun_cpmg_J(B_c):
    return 1/2 - 1/2*A_fit*j0(B_c)   

fitguess = np.array([G_tot,0.02])
fithold = np.array([True,False])
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
datas = []
errors = []
stds = []
count_vars = []
count_avgs = []

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
    #int_t[i] = 2e-6*data_p["squeeze_arm_t"]  #total interaction time in secs
    J1ks[i] = data_p['raman_J1kHz'] 
    Ncal = data_p['det_phtperionperms']
    k[i] = bm-dm  # phtns per N atoms
    N[i] = k[i]/(det_t*1e-3)/Ncal
    t = data_p['sf_fitParam_tpi']*1e-6
    u_ang = 180-2*(hf.get_ionProp_value('raman%wp%raman_wp_upper_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_lower_home_offset'))
    l_ang = 2*(hf.get_ionProp_value('raman%wp%raman_wp_lower_acssn_a0')-hf.get_ionProp_value('raman%wp%raman_wp_upper_home_offset'))
    k_ion = k[i]/N[i]   
    G_tot = 0.5*( hf.get_ionProp_value('raman%align%spontem%G_du') +
        hf.get_ionProp_value('raman%align%spontem%G_ud') +
        hf.get_ionProp_value('raman%align%spontem%G_el') )
    mVpp = hf.get_ionProp_value('raman%force_sensing%drive_amp_mVpp')
    z = (mVpp*1e-9)/2
    
    # load experiment data, but ignore as I want to do cacls from the
    # raw data here
    for file in os.listdir(os.getcwd()):
        if file.startswith("histData.") and file.endswith(".csv"):
            file_name = file
    props = hf.parse_raw_counts_data(file_name)[1]
    T = props['arm_t']*1e-6
    tau = np.mean(props['arm_t'])*1e-6*tau_fac
    mu  = props['raman_df']*1e3

    data_name = [x for x in files if "_data.csv" in x][0]
    file_name, data = hf.get_gen_csv(data_name, skip_header=True)
    odf_pwr[i] = np.array(data.T[0][0:],dtype='float')
    counts[i] = np.array(data.T[1][0:],dtype='float')
    trials = np.array(data.T[3][0:],dtype='float')
    pts = len(trials)
    err[i] = np.array(data.T[2][0:],dtype='float')/np.sqrt(trials)
    
#theory params
    odf_th_pwr = np.linspace(min(odf_pwr[i]),max(odf_pwr[i]),pts)
    ACSS_l = odf_th_pwr*3.4e4
    ACSS_u = odf_th_pwr*3.4e4
    U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
    U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
    U = .5*(U_u+U_l)
    DWF = 0.86
    B_c_D = DWF*U*dk*z*tau
    gamma = odf_th_pwr*G_tot
    A = np.exp(-tau*gamma)
    
#    bck = (1-A)/2
    
    bp = ((counts[i]-dm)/k[i])
    
    if i is 1:
        label = '{:g} nm displacement at {:g} kHz'.format(z*1e9,mu*1e-3)
        
    # Load histgram
    if raw is True:
        # Load histgram
        data_name = [x for x in files if "_raw.csv" in x][0]
        hdata = np.genfromtxt(data_name, delimiter=",", dtype='float')
        reps = np.shape(hdata)[1]
        pts = np.shape(hdata)[0]
        hdata_re = np.reshape(hdata,(pts,1,reps))
        bp_raw_avg = np.ones(int(pts))
        bp_raw_err = np.ones(int(pts))
        bp_raw_std = np.ones(int(pts))
        counts_data_var = np.ones(int(pts))
        counts_data_avg = np.ones(int(pts))

        for j,row in enumerate(hdata_re):
            counts_data = hf.parse_raw_counts(row)
            counts_data_avg[j] = np.mean(counts_data)
            counts_data_var[j] = np.var(counts_data)
            bp_raw = ((counts_data-dm)/k[i])
            bp_raw_avg[j] = np.mean(bp_raw)
            bp_raw_err[j] = np.std(bp_raw)/sqrt(np.size(bp_raw))
            bp_raw_std[j] = np.std(bp_raw)
        
    if i is 0:
        label = 'Background ' + fn[:-7]

        popt,perr=pt.plot_fit(odf_pwr[0],bp_raw_avg,bck_fit,fitguess,yerr=bp_raw_err,data_label=label,fit_label='fit',hold=fithold,fmt_data='o',fmt_fit='--')
        A_fit = np.exp(-tau*gamma)*np.exp(-0.5*popt[0])
        bck = (1-A_fit)/2
        
        plt.show()
        plt.close()
    datas += [bp_raw_avg]
    errors += [bp_raw_err]
    stds += [bp_raw_std]
    count_vars += [counts_data_var]    
    count_avgs += [counts_data_avg]
    os.chdir(base_path)

    if i is 1:
        sub = (datas[1] - bck)/A_fit
        diffs = 1-2*sub
        theta_max =  np.ones(int(pts))
        for k,diff in enumerate(diffs):
            if diff > 1:
                diff = 1
            theta_max[k] = brentq(root,a,b,args=(diff,))
        theta_max2 = (theta_max)**2
            
        theta_max_m = np.linspace(min(theta_max),max(theta_max),pts)
        plt.errorbar(odf_pwr[i],bp_raw_avg,yerr=bp_raw_err,linestyle = '', marker='o',label=label)

 
delta_F = sqrt(stds[0]**2+stds[1]**2)/A_fit
F_deriv = 0.25*j1(sqrt(theta_max2))/sqrt(theta_max2)
delta_theta2 = delta_F/F_deriv
S_N = theta_max2/delta_theta2

plt.plot(odf_pwr[i],fun_cpmg_J(B_c_D),'--',label='Theory for CPMG peak with DWF = {:g}'.format(DWF))     
plt.plot(odf_pwr[i],(1-A_fit)/2,'--',label=r'$\exp(-\tau\Gamma_{tot})$' + r' with $\Gamma_{tot} = $' + r' {:g}'.format(G_tot) + ' plus bck from mag fluc') 
plt.title(r'8 pi CPMG, 24 ms total time '+ fn[:-7])
plt.xlabel('Fractional ODF Power')
plt.ylabel('Bright Fraction')
plt.ylim(0,1)
#plt.xlim(0,.5)
plt.legend(title='N = {:.0f}, wr = 178 kHz'.format(N[i]), loc=(1,0))
plt.show()
plt.close()

plt.plot(odf_pwr[i],theta_max2,'o', label='{:g} nm displacement at {:g} kHz'.format(z*1e9,mu*1e-3)) 
plt.title(r'8 pi CPMG, 24 ms total time '+ fn[:-7])
plt.xlabel('Fractional ODF Power')
plt.ylabel(r'$\theta_{max}^2$ (rad)')
plt.legend(title='N = {:.0f}, wr = 178 kHz'.format(N[i]), loc=(1,0))
plt.show()
plt.close()

plt.plot(odf_pwr[i],S_N,'o', label='{:g} nm displacement at {:g} kHz'.format(z*1e9,mu*1e-3)) 

sig_proj = 1/sqrt(N[1])*sqrt(datas[1]*(1-datas[1]))
bck_proj = 1/sqrt(N[1])*sqrt(0.25*(1-A_fit**2))
delta_th = (A_fit**2)/8*(1+j0(2*theta_max_m)-2*j0(theta_max_m)**2)
delta_J_m = 2*A_fit**(-1)*sqrt(sig_proj**2 + bck_proj**2 + delta_th)
delta_th_m = delta_J_m/(j1(theta_max_m))
S_N_m = theta_max_m/delta_th_m

plt.plot(odf_pwr[i],S_N_m/2,'--')

plt.title(r'8 pi CPMG, {:.1f} ms total time '.format(tau*1e3)+ fn[:-7])
plt.xlabel('Fractional ODF Power')
plt.ylabel('Signal to Noise')
plt.ylim(0,1.5)
plt.legend(title='N = {:.0f}, wr = 176 kHz'.format(N[i]), loc=(1,0))
plt.show()
plt.close()

#sens = 1e24*F*N[1]*sqrt(200*(2*tau+drive_t))/S_N
#plt.plot(odf_pwr[i],sens,'o')
#
#plt.title(r'1 ms drive, 2 ms ODF '+ fn[:-4])
#plt.xlabel(r'$\theta_{max}$ (rad)')
#plt.ylabel('Amplitude sensitivity (yN per root Hz)')
##plt.ylim(0,1.5)
#plt.legend()
#plt.show()
#plt.close()

#plt.plot(odf_pwr[i],noise,'o') 
#plt.title(r'4 ms, 2 $\pi$ pulse CPMG  '+ fn[:-4])
#plt.xlabel('Fractional ODF Power')
#plt.ylabel('Noise (std dev of signal)')
#plt.legend()
#plt.show()
#plt.close()
#
#bck_est_proj = k_ion**2*N[0]*datas[0]*(1-datas[0])
#bck_est_mag = (k_ion*datas[0]*(1.4*N[0]))**2
#plt.plot(odf_pwr[i],stds[0],'o',label='Std dev of bright fraction') 
#plt.plot(odf_pwr[i],np.sqrt(bck_est_proj+mag_var)/k[0],'--',label='Projection noise limited (mag fluc added)') 
##plt.plot(odf_pwr[i],bck_est_mag,'--',label='Magnetic field noise limited') 
#plt.title(r'4 ms, 2 $\pi$ pulse CPMG  '+ bck_title)
#plt.xlabel('Fractional ODF Power')
#plt.ylabel('Background std dev (bright fraction)')
#plt.legend(loc=(1,0))
#plt.show()
#plt.close()