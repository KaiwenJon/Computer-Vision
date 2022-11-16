# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import scipy

def fit_fundamental(matches, normalize):
    pl_list = matches[:, 0:2]
    pr_list = matches[:, 2:4]
    if(normalize == True):
        pl_list, T1 = normalize_pts(pl_list)
        pr_list, T2 = normalize_pts(pr_list)
    # RANSAC
    num_iteration = 1000
    best_F = None
    best_residual = 100
    best_num_inlier = 100
    best_inl_res = 0
    best_inliers = None
    print("Number of matches: ", matches.shape[0])
    
    for iter in range(num_iteration):
        num_inliers = 0
        inlier_res = 0
        rand_idx = random.sample(range(pl_list.shape[0]), k=15)
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
            new_F = T1.transpose() @ new_F @ T2 
            # new_F = T2.transpose() @ new_F @ T1
        residual, inliers, inlier_res = get_residual(matches, new_F)

        if(residual < best_residual):
            best_residual = residual
            best_F = new_F
            best_num_inlier = inliers.shape[0]
            best_inl_res = inlier_res
            best_inliers = inliers
            print('residual: ', residual)
    print("Number of inliers: ", best_num_inlier, "\tAverage res error of inliers: ",  best_inl_res/best_num_inlier)
    print("Overall res: ", best_residual)
    return best_F, best_inliers
def get_residual(matches, F):
    pl = matches[:, 0:2]
    pr = matches[:, 2:4]
    pl_homo = np.concatenate((pl, np.ones((pl.shape[0], 1))), axis=1)
    pr_homo = np.concatenate((pr, np.ones((pr.shape[0], 1))), axis=1)
    inliers = []
    residual = 0
    inlier_res = 0
    for i in range(pl.shape[0]):
        dist = abs(pl_homo[i] @ F @ pr_homo[i].transpose())
        residual += dist
        if(dist < 0.1):
            inliers.append(matches[i, 0:4])
            inlier_res += dist
    inliers = np.array(inliers)
    residual = residual / matches.shape[0]
    return residual, inliers, inlier_res
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
# comment from here
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
F, _ = fit_fundamental(matches, normalize=True); # this is a function that you should write
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

# comment to here
## Camera Calibration
points_2d = np.loadtxt('MP4_part2_data/lab_matches.txt')
points_2d_1 = points_2d[:, 0:2]
points_2d_2 = points_2d[:, 2:4]
points_3d = np.loadtxt('MP4_part2_data/lab_3d.txt')

points_2d_lib = np.loadtxt('MP4_part2_data/library_matches.txt')
points_2d_lib_1 = points_2d_lib[:, 0:2]
points_2d_lib_2 = points_2d_lib[:, 2:4]

def camera_calibration(points_2d, points_3d):
    N = points_3d.shape[0]
    A = []
    for i in range(N):
        x = points_3d[i][0]
        y = points_3d[i][1]
        z = points_3d[i][2]
        u = points_2d[i][0]
        v = points_2d[i][1]
        row1 = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
        row2 = [00, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
        A.append(row1)
        A.append(row2)
    A = np.array(A)

    U, s, V = np.linalg.svd(A)
    P = V[len(V)-1].reshape(3, 4)
    P = P / P[2, 3]
    return P


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

P_1 = camera_calibration(points_2d_1, points_3d)
P_2 = camera_calibration(points_2d_2, points_3d)
points_3d_proj_1, residual_1 = evaluate_points(P_1, points_2d_1, points_3d)
points_3d_proj_2, residual_2 = evaluate_points(P_2, points_2d_2, points_3d)
print(P_1)
print(P_2)
print(residual_1)
print(residual_2)
# print(points_3d_proj)
# sq_dist = np.sqrt(np.sum((points_2d_1 - points_3d_proj)**2))
# print(sq_dist)
# print(residual)
def get_camera_centers(P):
    U, s, V = np.linalg.svd(P)
    center = V[len(V)-1]
    center /= center[-1]
    return center
## Camera Centers
P_lib1 = np.loadtxt('MP4_part2_data/library1_camera.txt')
P_lib2 = np.loadtxt('MP4_part2_data/library2_camera.txt')

center_lab_1 = get_camera_centers(P_1)
print(center_lab_1)

center_lab_2 = get_camera_centers(P_2)
print(center_lab_2)

center_lib_1 = get_camera_centers(P_lib1)
print(center_lib_1)

center_lib_2 = get_camera_centers(P_lib2)
print(center_lib_2)

## Triangulation

def triangulation(points_2d_1, points_2d_2, P_1, P_2):
    N = points_2d_1.shape[0]
    points_2d_1 = np.concatenate((points_2d_1, np.ones((N, 1))), axis = 1)
    points_2d_2 = np.concatenate((points_2d_2, np.ones((N, 1))), axis = 1)
    X_3d = []
    for i in range(N):
        x1_cross_P1 = np.array([[0, -points_2d_1[i,2], points_2d_1[i,1]], 
                                [points_2d_1[i,2], 0, -points_2d_1[i,0]], 
                                [-points_2d_1[i,1], points_2d_1[i,0], 0]])
        x2_cross_P2 = np.array([[0, -points_2d_2[i,2], points_2d_2[i,1]], 
                                [points_2d_2[i,2], 0, -points_2d_2[i,0]], 
                                [-points_2d_2[i,1], points_2d_2[i,0], 0]])

        x_cross_P = np.concatenate((x1_cross_P1 @ P_1, x2_cross_P2 @ P_2), axis=0)
        
        U, s, V = np.linalg.svd(x_cross_P)
        X = V[len(V)-1]
        X /= X[-1]
        # print(X)
        X_3d.append(X)
    X_3d = np.array(X_3d)
    return X_3d
X_3d = triangulation(points_2d_1, points_2d_2, P_1, P_2)
X_3d_lib = triangulation(points_2d_lib_1, points_2d_lib_2, P_lib1, P_lib2)
print(X_3d.shape)
print(X_3d)
print(X_3d_lib)
_, residual_1 = evaluate_points(P_1, points_2d_1, X_3d[:, 0:3])
_, residual_2 = evaluate_points(P_2, points_2d_2, X_3d[:, 0:3])
print("residual of reproj of triangulation: ",residual_1+residual_2)
import mpl_toolkits.mplot3d.axes3d as p3
def plot_3d(center1, center2, X_3d):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c='b', marker='o', alpha=0.6)
    ax.scatter(center1[0], center1[1], center1[2], c='r', marker='+')
    ax.scatter(center2[0], center2[1], center2[2], c='g', marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

plot_3d(center_lab_1, center_lab_2, X_3d)
plot_3d(center_lib_1, center_lib_2, X_3d_lib)

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    print(inliers.shape[0])
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')
class Matching():
    def __init__(self, img1, img2):
        self.img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        self.feature_matching()
        F, self.inliers = fit_fundamental(self.candidates, True)
        print(self.inliers.shape)
        print(self.candidates.shape)
        fig, ax = plt.subplots(figsize=(20,10))
        plot_inlier_matches(ax, self.img1, self.img2, self.inliers)
        plt.show()

    def feature_matching(self):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.gray1,None)
        kp2, des2 = sift.detectAndCompute(self.gray2,None)
        matches = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
        cross_check_matches = []
        des1_Love = np.argmin(matches, axis=1)
        des2_Love = np.argmin(matches, axis=0)
        for i in range(des1_Love.shape[0]):
            lover = des1_Love[i] # descriptor L likes index: lover the most
            if(des2_Love[lover] == i):
                cross_check_matches.append([i, lover])
        candidates = []
        for i, match in enumerate(cross_check_matches):
            score = matches[match[0]][match[1]]
            print(score)
            if(score > 30000):
                continue
            candidate = np.zeros(4)
            pointL = kp1[match[0]]
            pointR = kp2[match[1]]
            candidate[0:2] = pointL.pt
            candidate[2:4] = pointR.pt 
            candidates.append(candidate)
        candidates = np.array(candidates)
        self.candidates = candidates

        print(self.candidates.shape)



if __name__ == '__main__':
    img1 = cv2.imread('MP4_part2_data/gaudi1.jpg')
    img2 = cv2.imread('MP4_part2_data/gaudi2.jpg')
    matcher = Matching(img1, img2)