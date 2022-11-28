import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import time

def showNormImage(image):
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    # print(normG)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', norm)
    cv2.waitKey(0)
img1 = cv2.imread("./stereo_data/tsukuba1.jpg", 0)
img2 = cv2.imread("./stereo_data/tsukuba2.jpg", 0)

# img1 = cv2.imread("./stereo_data/moebius1.png", 0)
# img2 = cv2.imread("./stereo_data/moebius2.png", 0)
# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21)
# disparity = stereo.compute(img1,img2)
# print(np.max(disparity))
# print(np.min(disparity))
# plt.figure()
# plt.imshow(disparity,'gray')
# plt.show()
# a = 50
# b = 150
# c = 50
# d = 150
# img1 = img1[a:b, c:d]
# img2 = img2[a:b, c:d]
h, w = img1.shape[0:2]
disparity_map = np.zeros((h,w))
window_size = 3 # i,j +- size
# pad image
# img1 = cv2.copyMakeBorder(img1, window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT, value=0)
# img2 = cv2.copyMakeBorder(img2, window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT, value=0)
# pad_h, pad_w = img1.shape[0:2]
disparity_map = np.zeros((h,w))
numDisparites = 16
# cv2.imshow("img", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
x =[]
y = []
y2= []
start = time.time()
for i in range(window_size, h-window_size):
    for j in range(window_size, w-window_size):
        search_window = img1[i-window_size:i+window_size, j-window_size:j+window_size]
        best_pix = 0
        best_cost = float("inf")
        for k in range(j-numDisparites, j):
            if(k < window_size or k+window_size > w):
                continue
            compare_window = img2[i-window_size:i+window_size, k-window_size:k+window_size]
            # cost = np.sum((search_window-compare_window)**2)
            cost = np.sum(np.abs(np.subtract(search_window, compare_window, dtype=np.float64)))
            # cost = np.sum(((search_window/np.linalg.norm(search_window)) * (compare_window/np.linalg.norm(compare_window))))

            if(cost < best_cost):
                best_cost = cost
                best_pix = k
                # print(np.sum(np.absolute(search_window-compare_window)))
                # print(i,j,k, cost)
            if(i == 156 and j == 222):
                x.append(k)
                y.append(cost)
                y2.append(np.sum((search_window-compare_window)**2))
        disparity = j - best_pix
        disparity_map[i][j] = disparity
end = time.time()
print("elapsed time", end-start)
plt.figure()
plt.imshow(disparity_map)
plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

plt.figure()
plt.plot(x, y, label="sad")
plt.plot(x, y2, label="ssd")
plt.legend()
plt.show()

