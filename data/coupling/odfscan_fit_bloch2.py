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
from scipy.special import j0,j1,jv
#from scicons import pi, hbar, m_Be, k_b,e

import hfGUIdata as hf
import plot_tools as pt
import qutip
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
    z = (mVpp*1e-9)/2

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
    
    for j,row in enumerate(hdata_re):
        counts_data = hf.parse_raw_counts(row)
        bp_raw = ((counts_data-dm)/k[i])
        bp_raw_avg[j] = np.mean(bp_raw)
        bp_raw_err[j] = np.std(bp_raw)/sqrt(np.size(bp_raw))
        bp_raw_std[j] = np.std(bp_raw)
    if i is 0:
        label = 'Background ' + fn[:-7]
        
        bck_raw = bp_raw
        bck = bp_raw_avg
        bck_err = bp_raw_std
        popt,perr=pt.plot_fit(odf_pwr[0],bck,bck_fit,fitguess,yerr=bp_raw_err,data_label=label,fit_label='fit',show_fit_param=False,hold=fithold,fmt_data='o',fmt_fit='--')

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
        plt.errorbar(odf_pwr[i],bp_raw_avg,yerr=bp_raw_err,linestyle = '', marker='o',label=label)     

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
        plt.plot(odf_th_pwr,fun_J(theta_max_th),'--',label='Theory')    
        
        subs = (bp_raw_avg- bck)/A_exp
        diffs = 1-2*subs
        theta_max =  np.ones(int(len(bp_raw_avg)))
        for l,diff in enumerate(diffs):
            if l is 25:
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
plt.xlabel(r'$U/U_{max}$')
plt.ylabel(r'$<P_{\uparrow}>$')
plt.ylim(0,0.6)
#plt.legend(title='N = {:.0f}, wr = 180 kHz'.format(N[i]), loc=(1,0))

ACSS_l = odf_pwr[1]*3.4e4
ACSS_u = odf_pwr[1]*3.3e4
U_u = 2*pi*ACSS_u*2*(np.cos(u_ang*pi/180))**2
U_l = 2*pi*ACSS_l*2*(np.cos(l_ang*pi/180))**2
U = .5*(U_u+U_l)
Z_c = theta_max/(DWF*U*tau*dk)*1e9 
theta_max_err = (2/A_exp)*sqrt(bp_raw_std**2 + bck_err**2)/abs(j1(theta_max))
Z_c_err = theta_max_err/(DWF*U*tau*dk)*1e9  
S_N = theta_max/theta_max_err/2
#plt.axes([.2, .68, .625, .2])
plt.axes([.55, .27, .3, .25])
#plt.show()
#plt.close()
plt.errorbar(odf_pwr[1][3:],Z_c[3:],yerr=Z_c_err[3:],linestyle = '', marker='.')
Z_c = np.full_like(odf_pwr[1],z*1e9) 
#plt.plot(odf_pwr[1],Z_c,'r--')  
#plt.axes([.55, .27, .3, .17])
#plt.plot(odf_th_pwr,theta_max_th)
plt.yticks(np.arange(0, 2, 1))
plt.xlabel(r'$U/U_{max}$')
plt.ylabel(r'$Z_{c}$ (nm)')
#plt.ylim(0,1.5)

plt.show()
plt.close()

import numpy as np
from qutip import Bloch

b1 = Bloch()
b1.zlabel = ['','']
b1.ylabel = ['','']
b1.xlabel = ['','']
b1.font_size = 30
b1.point_marker = ['o']
b1.point_color = ['r']
b1.point_size = [10]
vec_num = 30
thetas = theta_max_th

for j, theta_max in enumerate(thetas):
    x = np.array([[cos(theta_max*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
    y = np.array([[sin(theta_max*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
    z = np.zeros((vec_num,1))
    
    x_j = j0(theta_max)
    xp = [x_j*cos(th) for th in np.linspace(0, pi/2, 10)]
    yp = np.zeros(10)
    zp = [x_j*sin(th) for th in np.linspace(0, -pi/2, 10)]
    
#    b1.add_points([xp,yp,zp])
#    b1.add_vectors(([0,0,-x_j],[x_j,0,0]))
    b1.add_vectors([x_j,0,0])
    b1.vector_color = ['r','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b']
    b1.vector_width = 1
    
    vecs = np.concatenate((x,y,z),axis=1)
    b1.add_vectors(vecs)
    b1.show()
    
    print(theta_max,odf_th_pwr[j])
    b1.clear()
    b1 = Bloch()
    b1.zlabel = ['','']
    b1.ylabel = ['','']
    b1.xlabel = ['','']
    b1.font_size = 30
    b1.point_marker = ['o']
    b1.point_size = [10]
    b1.point_color = ['r']
    b1.frame_alpha = 0.001