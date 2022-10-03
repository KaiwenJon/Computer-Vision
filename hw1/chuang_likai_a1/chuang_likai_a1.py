import math
import cv2
import numpy as np
import time

class imageAlign():
    def __init__(self, image):
        self.image = image
        height, width= image.shape[0:2]
        self.imageB = image[0:round(height/3),:, 1]
        self.imageG = image[round(height/3): round(2*height/3), :, 1]
        self.imageR = image[round(2*height/3):, :, 1]
        # padding
        if(self.imageB.shape[0] > self.imageG.shape[0]):
            self.imageG = np.concatenate((self.imageG, np.zeros(shape=(self.imageB.shape[0] - self.imageG.shape[0], width), dtype = 'uint8')), axis=0)
        elif(self.imageG.shape[0] > self.imageB.shape[0]):
            self.imageB = np.concatenate((self.imageB, np.zeros(shape=(self.imageG.shape[0] - self.imageB.shape[0], width), dtype = 'uint8')), axis=0)
        if(self.imageB.shape[0] > self.imageR.shape[0]):
            self.imageR = np.concatenate((self.imageR, np.zeros(shape=(self.imageB.shape[0] - self.imageR.shape[0], width), dtype = 'uint8')), axis=0)
        elif(self.imageG.shape[0] > self.imageB.shape[0]):
            self.imageB = np.concatenate((self.imageB, np.zeros(shape=(self.imageR.shape[0] - self.imageB.shape[0], width), dtype = 'uint8')), axis=0)
        # print(self.imageB.shape[0])
        # print(self.imageG.shape[0])
        # print(self.imageR.shape[0])
    def pyramids_align(self):
        def subPyramids(scale, iG0, jG0, iR0, jR0):
            if(scale < 1):
                self.iG = iG0
                self.jG = jG0
                self.iR = iR0
                self.jR = jR0
                self.alignedG = np.roll(self.imageG,[iG0,jG0],axis=(0,1))
                self.alignedR = np.roll(self.imageR,[iR0,jR0],axis=(0,1))
                return
            height, width = self.imageB.shape[0:2]
            imgB = cv2.resize(self.imageB, (round(width/scale), round(height/scale)))
            imgG = cv2.resize(self.imageG, (round(width/scale), round(height/scale)))
            imgR = cv2.resize(self.imageR, (round(width/scale), round(height/scale)))
            iG, jG = self.nccAlign(imgB, imgG, scale=scale, i0=iG0, j0=jG0, type="pyramid")
            iR, jR = self.nccAlign(imgB, imgR, scale=scale, i0=iR0, j0=jR0, type="pyramid")
            alignedG = np.roll(imgG,[iG,jG],axis=(0,1))
            alignedR = np.roll(imgR,[iR,jR],axis=(0,1))
            stacked_img = cv2.merge((imgB, alignedG, alignedR))
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', stacked_img)
            print("Press any key to continue to the next layer.")
            cv2.waitKey(0) 
            subPyramids(int(scale/2), iG, jG, iR, jR)
            return 
        subPyramids(8, 0, 0, 0, 0)

    def normal_align(self):
        iG, jG = self.nccAlign(self.imageB, self.imageG, scale=1, i0=0, j0=0, type="normal")
        self.alignedG = np.roll(self.imageG,[iG,jG],axis=(0,1))
        iR, jR = self.nccAlign(self.imageB, self.imageR, scale=1, i0=0, j0=0, type="normal")
        self.alignedR = np.roll(self.imageR,[iR,jR],axis=(0,1))
    
    def nccAlign(self, image1, image2, scale, i0, j0, type):
        def ncc(a, b):
            h, w = a.shape[0:2]
            # a = a[round(h/10): round(9*h/10), round(w/10): round(9*w/10)]
            # b = b[round(h/10): round(9*h/10), round(w/10): round(9*w/10)]
            a = a[round(2*h/10): round(8*h/10), round(2*w/10): round(8*w/10)]
            b = b[round(2*h/10): round(8*h/10), round(2*w/10): round(8*w/10)]
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', a/np.linalg.norm(a)*255)
            # print("Press any key to continue to the next layer.")
            # cv2.waitKey(0)
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', b/np.linalg.norm(b)*255)
            # print("Press any key to continue to the next layer.")
            # cv2.waitKey(0)  
            return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))
        image1 = image1-image1.mean(axis=0)
        image2 = image2-image2.mean(axis=0)
        maxScore = -1
        besti, bestj = 0, 0
        if(type == "normal"):
            s = 15
        elif(type == "pyramid"):
            s = 2
        for i in range(2*i0-scale*s, 2*i0+scale*s):
            for j in range(2*j0-scale*s, 2*j0+scale*s):
                score = ncc(image1, np.roll(image2,[i,j],axis=(0,1)))
                if(score > maxScore):
                    maxScore = score
                    besti, bestj = i, j 
        print("on scale:", scale,",i center:",2*i0, ", j center:",2*j0, ", i range:", 2*i0-scale*s, " to ", 2*i0+scale*s, " j range: ",2*j0-scale*s, " to ", 2*j0+scale*s)
        print(besti, bestj)
        return besti, bestj

if __name__ == "__main__":
    # image = cv2.imread("./data/00125v.jpg")
    image = cv2.imread("./data_hires/01657u.tif")
    aligner = imageAlign(image)
    start = time.time()
    ##NOTE Use normal_align() for low-resolution image
    aligner.normal_align()

    ##NOTE Use pyramids_align() for high-resolution image
    # aligner.pyramids_align()
    end = time.time()
    stacked_img = cv2.merge((aligner.imageB, aligner.alignedG, aligner.alignedR))
    # stacked_img = cv2.merge((aligner.imageB, aligner.imageG, aligner.imageR))
    

    height, width = stacked_img.shape[0:2]
    stacked_img = stacked_img[round(height/20):round(19*height/20), round(width/20):round(19*width/20), :]
    print("Done")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', stacked_img)
    cv2.waitKey(0) 
    # cv2.imwrite("./output/weird_00125v.jpg", stacked_img)
    print(end-start)