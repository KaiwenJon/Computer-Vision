import numpy as np
import cv2

def showNormImage(image):
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    # print(normG)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', norm)
    cv2.waitKey(0)
img1 = cv2.imread("./stereo_data/tsukuba1.jpg")
img2 = cv2.imread("./stereo_data/tsukuba2.jpg")

a = 50
b = 150
c = 50
d = 150
img1 = img1[a:b, c:d]
img2 = img2[a:b, c:d]
h, w = img1.shape[0:2]
disparity_map = np.zeros((h,w))
window_size = 5 # i,j +- size
# pad image
# img1 = cv2.copyMakeBorder(img1, window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT, value=0)
# img2 = cv2.copyMakeBorder(img2, window_size, window_size, window_size, window_size, cv2.BORDER_CONSTANT, value=0)
# pad_h, pad_w = img1.shape[0:2]
disparity_map = np.zeros((h,w))
# cv2.imshow("img", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for i in range(window_size, h-window_size):
    for j in range(window_size, w-window_size):
        search_window = img1[i-window_size:i+window_size, j-window_size:j+window_size]
        best_pix = 0
        best_cost = 10000000
        for k in range(window_size, w-window_size):
            compare_window = img2[i-window_size:i+window_size, k-window_size:k+window_size]
            cost = np.sum((search_window - compare_window)**2)
            if(cost < best_cost):
                best_cost = cost
                best_pix = k
                # print(i,j,k, cost)
        disparity = best_pix - j
        if(disparity > 10 or disparity < -10):
            disparity = disparity_map[i][j-1]
        disparity_map[i][j] = disparity
disparity_map[disparity_map==0] = np.min(disparity_map)
print(np.max(disparity_map))
print(np.min(disparity_map))
norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
norm = norm.astype(np.uint8)
total_num_disp = 50
norm = (norm/(255/total_num_disp)).astype(np.int64)
norm = (norm * (255/total_num_disp)).astype(np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', norm)
cv2.waitKey(0)
# showNormImage(disparity_map)
# cv2.imshow("img", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

