import cv2
import numpy as np
from sympy import *

import matplotlib.pyplot as plt

measure_M = np.loadtxt('./factorization_data/measurement_matrix.txt')
num_views = int(measure_M.shape[0]/2)
num_feature = measure_M.shape[1]
centroids = np.mean(measure_M, axis = 1)
centroids = np.reshape(centroids, (centroids.shape[0], 1))
measure_M_subCen = np.subtract(measure_M , centroids)
# print(measure_M_subCen)

U, s, V = np.linalg.svd(measure_M_subCen)
s = np.diag(s)
U1 = U[:, 0:3]
s1 = s[0:3, 0:3]
V1 = (V)[0:3, :]
s1_sqrt = np.sqrt(s1)

M = U1 @ s1_sqrt
S = s1_sqrt @ V1
print(U1.shape, s1.shape, V1.shape)

# assume QQT has 9 entries q1, q2, ..., q9
# q1, q2, q3, q4, q5, q6, q7, q8, q9 = symbols('q1, q2, q3, q4, q5, q6, q7, q8, q9')
# a1, a2, a3, a4, a5, a6 = symbols('a1, a2, a3, a4, a5, a6')
# Q = Matrix([[q1, q2, q3],
#             [q4, q5, q6],
#             [q7, q8, q9]])
# print(M)
myA = []
myB = []
for i in range(num_views):
    A = M[i*2 : i*2+2, :]
    a1, a2, a3, a4, a5, a6 = np.reshape(A, (1, 6))[0]
    r1 = [a1*a1, a1*a2, a1*a3, a1*a2, a2*a2, a2*a3, a1*a3, a3*a2, a3*a3]
    r2 = [a1*a4, a1*a5, a1*a6, a2*a4, a2*a5, a2*a6, a3*a4, a3*a5, a3*a6]
    r3 = [a1*a4, a2*a4, a3*a4, a1*a5, a2*a5, a3*a5, a1*a6, a2*a6, a3*a6]
    r4 = [a4*a4, a4*a5, a4*a6, a4*a5, a5*a5, a5*a6, a4*a6, a6*a5, a6*a6]
    myA.append(r1)
    myA.append(r2)
    myA.append(r3)
    myA.append(r4)
    myB.append([1])
    myB.append([0])
    myB.append([0])
    myB.append([1])
    # myB.append([1])
    # myB.append([1])
    # myB.append([0])
myA = np.array(myA)
myB = np.array(myB)
solution = np.linalg.lstsq(myA, myB)
QQT = solution[0].reshape((3, 3))
print("QQT:")
print(QQT)
Q = np.linalg.cholesky(QQT)
# Q = np.eye(3,3)
print("Q:")
print(Q)
print(Q@Q.T)
new_M = M @ Q
new_S = np.linalg.inv(Q) @ S
X = new_S[0, :]
Y = new_S[1, :]
Z = new_S[2, :]
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_proj_type('persp',focal_length=0.2) 
ax.scatter(X, Y, -Z, c='b', marker='o', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

ind = []
res = []
for i in range(num_views):
    x = measure_M_subCen[i*2, :]
    y = measure_M_subCen[i*2+1, :]
    proj_x = new_M[i*2,:] @ new_S
    proj_y = new_M[i*2+1,:] @ new_S
    euc_dist = np.sum((x-proj_x)**2 + (y-proj_y)**2)
    ind.append(i+1)
    res.append(euc_dist)
    print(euc_dist)
plt.figure()
# plt.scatter(x, y, c='b')
# plt.scatter(proj_x, proj_y, c='r')
plt.plot(ind, res)
plt.show()

plt.figure()
i = 10
x = measure_M_subCen[i*2, :]
y = measure_M_subCen[i*2+1, :]
proj_x = new_M[i*2,:] @ new_S
proj_y = new_M[i*2+1,:] @ new_S
plt.scatter(x, y, c='b')
plt.scatter(proj_x, proj_y, c='r')
plt.show()