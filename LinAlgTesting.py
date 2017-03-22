import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from math import sqrt

"""a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1, -1], [1, 1]])

print(a)
print(b)

c = b @ a
print(c)"""

mat = scipy.io.loadmat('face_detect.mat')
#type(mat)
face_data = mat['faces_train']
sh = face_data.shape
#print(sh)

f_re = np.reshape(face_data,(sh[0]*sh[1], sh[2]))
f_mean  = np.expand_dims(np.mean(f_re, axis=0), axis=0)
#print(f_mean.shape)
f_mean_centered = f_re - (np.ones((sh[0]*sh[1],1)) @ f_mean)/sqrt(sh[0]*sh[1])
#print(f_mean_centered)
f_cov = (np.transpose(f_mean_centered) @ f_mean_centered) / (sh[0]*sh[1])
[lambs, evecs] = np.linalg.eig(f_cov)
evecs2 = f_mean_centered @ evecs
evecs2 = evecs2 / np.linalg.norm(evecs2, axis=0)
print('evecs2: ')
print(evecs2.shape)

print((evecs2.T @ evecs2))

"""print('lambs: ')
print(lambs)
print('u: ')
print(u.shape)

print(f_cov)
print(u)"""



"""f_ave_data = f_re @ (np.ones((sh[2],1)))/sh[2]
f_ave = np.reshape(f_ave_data,(sh[0], sh[1]))
scipy.misc.imsave('outfile.png', f_ave)

## Test Eig and Covarience
#[v, d] = np.linalg.eig(f_re)
print(v)
print(d)"""
