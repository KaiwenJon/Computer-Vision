import math
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class imageAlign():
    def __init__(self, image):
        self.imageB_raw, self.imageG_raw, self.imageR_raw = self.divide_and_padding(image)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image_preprocessed = cv2.filter2D(image, ddepth=-1, kernel=kernel)
        self.imageB, self.imageG, self.imageR = self.divide_and_padding(image_preprocessed)

    def divide_and_padding(self, image):
        height, width= image.shape[0:2]
        imageB = image[0:round(height/3),:, 1]
        imageG = image[round(height/3): round(2*height/3), :, 1]
        imageR = image[round(2*height/3):, :, 1]
        # padding
        if(imageB.shape[0] > imageG.shape[0]):
            imageG = np.concatenate((imageG, np.zeros(shape=(imageB.shape[0] - imageG.shape[0], width), dtype = 'uint8')), axis=0)
        elif(imageG.shape[0] > imageB.shape[0]):
            imageB = np.concatenate((imageB, np.zeros(shape=(imageG.shape[0] - imageB.shape[0], width), dtype = 'uint8')), axis=0)
        if(imageB.shape[0] > imageR.shape[0]):
            imageR = np.concatenate((imageR, np.zeros(shape=(imageB.shape[0] - imageR.shape[0], width), dtype = 'uint8')), axis=0)
        elif(imageG.shape[0] > imageB.shape[0]):
            imageB = np.concatenate((imageB, np.zeros(shape=(imageR.shape[0] - self.imageB.shape[0], width), dtype = 'uint8')), axis=0)
        return imageB, imageG, imageR

    def align(self):
        iG, jG = self.fft_align(self.imageB, self.imageG)
        print(iG, jG)
        self.alignedG = np.roll(self.imageG_raw,[iG,jG],axis=(0,1))
        iR, jR = self.fft_align(self.imageB, self.imageR)
        print(iR, jR)
        self.alignedR = np.roll(self.imageR_raw,[iR,jR],axis=(0,1))
    def fft_align(self, c1, c2):
        height, width = c1.shape
        c1 = c1[round(1*height/10): round(9*height/10), round(1*width/10): round(9*width/10)]
        c2 = c2[round(1*height/10): round(9*height/10), round(1*width/10): round(9*width/10)]
        fft1 = np.fft.fft2(c1)
        fft1 = np.fft.fftshift(fft1)
        fft2 = np.fft.fft2(c2)
        fft2 = np.fft.fftshift(fft2)
        # mag = 20*np.log(abs(self.fftB)).astype(np.uint8)
        # print(mag)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', mag)
        # cv2.waitKey(0)
        product = fft1 * np.conjugate(fft2)
        product = np.fft.ifftshift(product)
        ifft =  np.fft.ifft2(product)
        ifft = np.abs(ifft)
        h, w = ifft.shape
        search_range = ifft[0:round(h/2), 0:round(w/2)]
        i, j = np.unravel_index(search_range.argmax(), search_range.shape)
        normalizedImg = np.zeros((h,w))
        cv2.normalize(ifft, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', normalizedImg.astype(np.uint8))
        cv2.waitKey(0)
        return i, j

if __name__ == "__main__":
    # image = cv2.imread("./data/01112v.jpg")
    image = cv2.imread("./data_hires/01047u.tif")
    aligner = imageAlign(np.array(image))
    aligner.align()
    stacked_img = cv2.merge((aligner.imageB_raw, aligner.alignedG, aligner.alignedR))
    height, width = stacked_img.shape[0:2]
    stacked_img = stacked_img[round(height/20):round(19*height/20), round(width/20):round(19*width/20), :]
    # height, width = stacked_img.shape[0:2]
    # stacked_img = stacked_img[round(height/20):round(19*height/20), round(width/20):round(19*width/20), :]
    print("Done")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', stacked_img)
    cv2.waitKey(0) 
    cv2.imwrite("./output/weird_00125v.jpg", stacked_img)