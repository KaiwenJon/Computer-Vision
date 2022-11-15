# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

def fit_fundamental(matches, normalize):
    pl_list = matches[:, 0:2]
    pr_list = matches[:, 2:4]
    if(normalize == True):
        pl_list, T1 = normalize_pts(pl_list)
        pr_list, T2 = normalize_pts(pr_list)
    # RANSAC
    num_iteration = 100
    best_F = None
    best_residual = 100
    for iter in range(num_iteration):
        rand_idx = random.sample(range(pl_list.shape[0]), k=30)
        # rand_idx = range(50, 75)
        pl_rand = pl_list[rand_idx]
        pr_rand = pr_list[rand_idx]

        A = []
        for i in range(pl_rand.shape[0]):
            ul, vl = pl_rand[i]
            ur, vr = pr_rand[i]
            # ur, vr = pl_rand[i]
            # ul, vl = pr_rand[i]
            
            row = [ul*ur, ul*vr, ul, vl*ur, vl*vr, vl, ur, vr, 1]
            A.append(row)
        A = np.array(A)

        U, s, V = np.linalg.svd(A)
        F = V[len(V)-1].reshape(3, 3)
        F = F / F[2, 2] 

        # Enforce rank-2
        U, s, v = np.linalg.svd(F)
        new_s = np.diag(s)
        new_s[-1] = 0
        new_F = U @ new_s @ v
        if normalize:
            new_F = T1.transpose() @ F @ T2 
            # new_F = T2.transpose() @ F @ T1
        residual = get_residual(matches, new_F)
        if(residual < best_residual):
            best_residual = residual
            best_F = new_F
            print('residual: ', residual)
    return best_F
def get_residual(matches, F):
    pl = matches[:, 0:2]
    pr = matches[:, 2:4]
    pl_homo = np.concatenate((pl, np.ones((pl.shape[0], 1))), axis=1)
    pr_homo = np.concatenate((pr, np.ones((pr.shape[0], 1))), axis=1)

    residual = 0
    for i in range(pl.shape[0]):
        residual += abs(pl_homo[i] @ F @ pr_homo[i].transpose())

    residual = residual / matches.shape[0]
    return residual
def normalize_pts(pts):
    mean = np.mean(pts, axis=0)
    pts_x_centered = pts[:, 0] - mean[0]
    pts_y_centered = pts[:, 1] - mean[1]

    scale = np.sqrt(1 / (2 * len(pts)) * np.sum(pts_x_centered**2 + pts_y_centered**2))
    scale = 1 / scale

    transform = np.array([[scale, 0, -scale*mean[0]], 
                           [0, scale, -scale*mean[1]], 
                           [0, 0, 1]])
    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    normalized = (transform @ pts.transpose()).transpose()

    return normalized[:, 0:2], transform
##
## load images and match files for the first example
##

I1 = Image.open('MP4_part2_data/library1.jpg')
I2 = Image.open('MP4_part2_data/library2.jpg')
matches = np.loadtxt('MP4_part2_data/library_matches.txt'); 

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image: 
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
I3[:,:I1.size[0],:] = I1
I3[:,I1.size[0]:,:] = I2
print(type(I3), type(I1))
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(np.uint8))
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
plt.show()

##
## display second image with epipolar lines reprojected 
## from the first image
##

# first, fit fundamental matrix to the matches
F = fit_fundamental(matches, normalize=True); # this is a function that you should write
# M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
# L1 = np.matmul(F, M).transpose() # transform points from 
M = np.c_[matches[:,0:2], np.ones((N,1))]
L1 = np.matmul(M, F) # transform points from 
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(np.uint8))
ax.plot(matches[:,2],matches[:,3],  '+r')
ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
plt.show()


## Camera Calibration

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual


## Camera Centers


## Triangulation

