import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from math import sqrt

basis_size = 60

print("Loading and preparing training  data...")

mat = scipy.io.loadmat('face_detect.mat')

faces_train = mat['faces_train']
names_train = mat['names_train']
faces_test_easy = mat['faces_test_easy']
names_test_easy = mat['names_test_easy']
faces_test_hard = mat['faces_test_hard']
names_test_hard = mat['names_test_hard']
sh = faces_train.shape


f_re = np.reshape(faces_train,(sh[0]*sh[1], sh[2]))
f_mean  = np.expand_dims(np.mean(f_re, axis=0), axis=0)
#print(f_mean.shape)
#print((np.ones((sh[0]*sh[1],1)) @ f_mean).shape)
f_mean_centered = (f_re - (np.ones((sh[0]*sh[1],1)) @ f_mean))

print('Calculating eigensystem...')

f_cov = (f_mean_centered.T @ f_mean_centered) / (sh[1]*sh[0])
[lambs, evecs] = np.linalg.eig(f_cov)
[lambs, evecs] = np.real(lambs), np.real(evecs)

idx = lambs.argsort()[::-1]
eigenValues = lambs[idx]
evecs = evecs[:,idx]

print('Making face basis with ' + str(basis_size) + ' eigenvectors...')

evecs2 = f_mean_centered @ evecs
face_basis = evecs2 / np.linalg.norm(evecs2, axis=0)

faces_proj = f_re.T @ face_basis

faces_predictor = faces_proj[:,:basis_size]

#face_data_1 = np.expand_dims(faces_predictor[index,:],axis=0) @ face_basis[:,:basis_size].T
#face_1 = np.reshape(face_data_1, (sh[0], sh[1]))
#scipy.misc.imsave('FaceTest1.png', face_1)

print('Done creating basis, loading test image...')

##################################################################################



input_face_index = 8

test_faces = faces_test_hard
test_names = names_test_hard

rotated_test_names = [['\x00' for i in range(len(test_names))] for j in range(max([len(test_names[i]) for i in range(len(test_names)-1)]))]

for i in range(len(test_names)):
    for j in range(len(test_names[i])):
        rotated_test_names[j][i] = test_names[i][j]

new_test_names = [""] * len(rotated_test_names)
        
for i in range(len(rotated_test_names)):
    for j in range(len(rotated_test_names[i])):
        if rotated_test_names[i][j] == '\x00':
            break
        else:
            new_test_names[i] += rotated_test_names[i][j]

test_names = new_test_names

train_names = names_train

rotated_train_names = [['\x00' for i in range(len(train_names))] for j in range(max([len(train_names[i]) for i in range(len(train_names)-1)]))]

for i in range(len(train_names)):
    for j in range(len(train_names[i])):
        rotated_train_names[j][i] = train_names[i][j]

new_train_names = [""] * len(rotated_train_names)
        
for i in range(len(rotated_train_names)):
    for j in range(len(rotated_train_names[i])):
        if rotated_train_names[i][j] == '\x00':
            break
        else:
            new_train_names[i] += rotated_train_names[i][j]

train_names = new_train_names


tsh = test_faces.shape
test_faces = np.reshape(test_faces, (tsh[0]*tsh[1], tsh[2]))

input_face = np.expand_dims(test_faces[:,input_face_index], axis=0)

print('Calculating test eigensystem...')

input_face_eig = input_face @ face_basis[:,:basis_size]

print('Classifying face...')

eigen_weight_diffs = (np.ones((sh[2],1)) @ input_face_eig) - faces_predictor

dists = np.linalg.norm(eigen_weight_diffs, axis=1)
face_guess_ind = np.argmin(dists)

print('Face real name: ' + test_names[input_face_index])
print('Face recognized name: ' + train_names[face_guess_ind])


