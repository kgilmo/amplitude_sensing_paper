# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:47:58 2017

@author: kag3
"""
import numpy as np
from numpy import pi
from qutip import Bloch
from scipy.special import j0

b1 = Bloch()
b1.zlabel = ['','']
b1.ylabel = ['','']
b1.xlabel = ['','']
b1.font_size = 30
b1.frame_alpha = 0.001   

vec_num = 1
theta = 1.41050139732

x = np.array([[np.cos(th)] for th in np.linspace(-theta, theta, vec_num)])
y = -np.array([[np.sin(th)] for th in np.linspace(-theta, theta, vec_num)])
z = np.zeros((vec_num,1))



b1.vector_color = ['r','r','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b']
b1.vector_width = 3

vecs = np.concatenate((x,y,z),axis=1)
b1.add_vectors(vecs)
b1.show()