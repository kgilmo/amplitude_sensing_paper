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
theta = pi/4

#x = np.array([[cos(theta*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
#y = np.array([[sin(theta*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
#z = np.zeros((vec_num,1))


# after final pi/2
y = np.array([[sin(theta*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
z = -np.array([[cos(theta*cos(phi))] for phi in np.linspace(0, pi, vec_num)])
x = np.zeros((vec_num,1))


x_j = j0(theta)
xp = [x_j*cos(th) for th in np.linspace(0, pi/2, 10)]
yp = np.zeros(10)
zp = [x_j*sin(th) for th in np.linspace(0, -pi/2, 10)]

#    b1.add_points([xp,yp,zp])
#    b1.add_vectors(([0,0,-x_j],[x_j,0,0]))
b1.clear()
b1 = Bloch()
b1.zlabel = ['','']
b1.ylabel = ['','']
b1.xlabel = ['','']
b1.font_size = 30
b1.point_marker = ['o']
b1.point_size = [10]
b1.point_color = ['r']
#b1.frame_alpha = 0.001   
b1.add_vectors([0,0,-x_j])
b1.vector_color = ['r','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b']
b1.vector_width = 1

vecs = np.concatenate((x,y,z),axis=1)
b1.add_vectors(vecs)
b1.show()
b1.save(dirc='bloch_sphere')