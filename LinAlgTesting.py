import numpy as np
import scipy.io

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1, -1], [1, 1]])

print(a)
print(b)

c = b @ a
print(c)

mat = scipy.io.loadmat('face_detect.mat')
type(mat)
face_data = mat['faces_train']
sh = face_data.shape
print(sh)

np.reshape(face_data,(sh[0]*sh[1], sh[2]))
