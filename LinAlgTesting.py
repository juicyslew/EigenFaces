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
index = 0

basis_size = 100

mat = scipy.io.loadmat('face_detect.mat')
#type(mat)
face_data = mat['faces_train']
sh = face_data.shape
#print(sh)

f_re = np.reshape(face_data,(sh[0]*sh[1], sh[2]))
f_mean  = np.expand_dims(np.mean(f_re, axis=0), axis=0)
#print(f_mean.shape)
print((np.ones((sh[0]*sh[1],1)) @ f_mean).shape)
f_mean_centered = (f_re - (np.ones((sh[0]*sh[1],1)) @ f_mean))

#print(f_mean_centered)
f_cov = (f_mean_centered.T @ f_mean_centered) / (sh[1]*sh[0])
[lambs, evecs] = np.linalg.eig(f_cov)
[lambs, evecs] = np.real(lambs), np.real(evecs)
print('evecs*evecs.T presorting: ')
print(evecs @ evecs.T)

###############################################################

idx = lambs.argsort()[::-1]
eigenValues = lambs[idx]
evecs = evecs[:,idx]
#to_be_sorted = np.hstack((np.expand_dims(lambs,1), evecs))

print('evecs*evecs.T sorting:')
print(evecs @ evecs.T)


#print(lambs.shape)
#print(evecs.shape)
evecs2 = f_mean_centered @ evecs
face_basis = evecs2 / np.linalg.norm(evecs2, axis=0)

#eig_face_1 = np.reshape(np.real(face_basis[:,1]), (sh[0], sh[1]))
#scipy.misc.imsave('eigenFace1.png', eig_face_1)
faces_proj = f_re.T @ face_basis
print('faces reshape data:')
print(f_re.shape)
print('faces basis: ')
print(face_basis.shape)
print('faces projection:')
print(faces_proj.shape)

faces_predictor = faces_proj[:,:basis_size]

face_data_1 = np.expand_dims(faces_predictor[index,:],axis=0) @ face_basis[:,:basis_size].T
face_1 = np.reshape(face_data_1, (sh[0], sh[1]))
scipy.misc.imsave('FaceTest1.png', face_1)

print(face_basis.T @ face_basis)


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
