from re import M
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy 
import skimage
import math
from PIL import Image
import random
def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')
class Stitching():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        self.feature_matching()
        H = self.ransac()
        self.warpImage(H)

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
            if(score > 10000):
                continue
            candidate = np.zeros(4)
            pointL = kp1[match[0]]
            pointR = kp2[match[1]]
            candidate[0:2] = pointL.pt
            candidate[2:4] = pointR.pt 
            candidates.append(candidate)
        candidates = np.array(candidates)
        self.candidates = candidates
    def ransac(self):
        iteration = 1000
        margin_thre = 0.5
        # inlier_thre = 10
        maxInlier = 0
        H_withMaxInlier = []
        for i in range(iteration):
            inlier_cnt = 0
            total_error = 0
            pick_idx = random.sample(range(self.candidates.shape[0]), 6)
            matches = []
            for idx in pick_idx:
                matches.append(self.candidates[idx, :])
            matches = np.array(matches)
            H = self.getHomography(matches)
            for j in range(self.candidates.shape[0]):
                xL, yL = self.candidates[j][0:2]
                xR, yR = self.candidates[j][2:4]
                predict = np.matmul(H, np.array([xL, yL, 1]))
                predict = predict / predict[2]
                predicted_xR, predicted_yR = predict[0:2]
                error = np.sqrt((xR-predicted_xR)**2 + (yR-predicted_yR)**2)
                total_error += error
                if(error < margin_thre):
                    inlier_cnt += 1
            if(inlier_cnt >= maxInlier):
                maxInlier = inlier_cnt
                print(maxInlier, total_error)
                H_withMaxInlier = H
                self.inliers = matches
        
        return H_withMaxInlier



    def getHomography(self, matches):
        A = []
        for i in range(matches.shape[0]):
            xL, yL = matches[i][0:2]
            xR, yR = matches[i][2:4]
            # print(xL, yL)
            # print(xR, yR)
            row1 = [0, 0, 0, xL, yL, 1, -yR*xL, -yR*yL, -yR]
            row2 = [xL, yL, 1, 0, 0, 0, -xR*xL, -xR*yL, -xR]
            A.append(row1)
            A.append(row2)
        A = np.array(A)
        U, s, V = np.linalg.svd(A)
        H = V[len(V)-1].reshape((3, 3))
        return H

    def warpImage(self, H):
        tp= skimage.transform.ProjectiveTransform(matrix = H)
        h, w = self.img2.shape[0:2]
        corners = np.array([[0, 0],
                            [0, h],
                            [w, 0],
                            [w, h]])
        warped_corners = tp(corners)
        print(warped_corners)
        corners = np.vstack((warped_corners, corners))
        cornerMin = np.min(corners, axis=0)
        cornerMax = np.max(corners, axis=0)
        print(cornerMin, cornerMax)
        output_shape = (cornerMax - cornerMin)
        output_shape = np.round(output_shape)
        output_shape = output_shape[::-1]
        print(output_shape)
        offset = skimage.transform.SimilarityTransform(translation=-cornerMin)
        # Find overlap
        image1 = skimage.transform.warp(self.img1, (tp + offset).inverse, output_shape=output_shape, cval=-1)
        image2 = skimage.transform.warp(self.img2, offset.inverse, output_shape=output_shape, cval=-1)
        overlap = np.multiply((image1 != -1), (image2 != -1)) # middle region is 1, otherwise it's all 0.


        image1 = skimage.transform.warp(self.img1, (tp + offset).inverse, output_shape=output_shape, cval=0)
        image2 = skimage.transform.warp(self.img2, offset.inverse, output_shape=output_shape, cval=0)

        # overlap += (overlap < 1).astype(int)
        merged = np.multiply(overlap == 0, (image1+image2)) + np.multiply(overlap, image1) 
        plt.figure()
        plt.imshow(merged)
        plt.show()

        # fig, ax = plt.subplots(figsize=(20,10))
        # plot_inlier_matches(ax, img1, img2, self.inliers)



if __name__ == '__main__':
    img1 = cv2.imread('./data/left.jpg')
    img2 = cv2.imread('./data/right.jpg')
    stither = Stitching(img1, img2)