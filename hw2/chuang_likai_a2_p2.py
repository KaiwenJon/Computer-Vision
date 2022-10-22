# Libraries you will find useful
import numpy as np
import cv2
import scipy 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import sympy  as sp
import math

class BlobDetector():
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray = self.gray.astype(np.float32)
        self.h, self.w = self.gray.shape
        # cv2.namedWindow('rawimage', cv2.WINDOW_NORMAL)
        # cv2.imshow('rawimage', image)
        # cv2.waitKey(0)
    
    def buildScaleSpaceGaussian(self):
        self.sigmaList = [2]
        for i in range(10):
            self.sigmaList.append(self.sigmaList[-1]*1.5)
        print(self.sigmaList)

        startedG = True
        startedScale = True
        for i in range(len(self.sigmaList)):
            sigma = self.sigmaList[i]
            kernel_size = round(6*sigma)
            if(kernel_size % 2 == 0):
                kernel_size += 1
            g = cv2.GaussianBlur(self.gray,(kernel_size, kernel_size), sigma)
            # g *= sigma**2
            # g = g**2
            if(startedG):
                gaussianStack = g
                startedG = False
            else:
                if(startedScale):
                    self.scaleSpace = g - gaussianStack
                    startedScale = False
                else:
                    newScale = g-gaussianStack[:,:,-1]
                    newScale = newScale**2
                    self.scaleSpace = np.dstack((self.scaleSpace, newScale))
                    self.showNormImage(self.scaleSpace[:, :, -1], sigma)
                gaussianStack = np.dstack((gaussianStack, g)) 
                # print(gaussianStack[:,:,-1])

    def buildScaleSpaceDownsample(self):
        self.sigmaList = [0.1, 0.6, 1]
        for i in range(10):
            self.sigmaList.append(self.sigmaList[-1]*1.5)
        print(self.sigmaList)

        started = True
        baseSigma = 2
        for i in range(len(self.sigmaList)):
            sigma = self.sigmaList[i]
            newWidth = round(self.w // (sigma/baseSigma))
            newHeight = round(self.h // (sigma/baseSigma))
            resizeG = cv2.resize(self.gray, (newWidth, newHeight))
            print(sigma, resizeG.shape)
            g = scipy.ndimage.gaussian_laplace(resizeG, sigma=baseSigma)
            # g *= baseSigma**2
            g = g**2
            g = cv2.resize(g, (self.w, self.h), interpolation = cv2.INTER_CUBIC)
            self.showNormImage(g, sigma)
            if(started):
                self.scaleSpace = g
                started = False
            else:
                self.scaleSpace = np.dstack((self.scaleSpace, g))

    def buildScaleSpace(self):
        self.sigmaList = [0.6, 1]
        for i in range(10):
            self.sigmaList.append(self.sigmaList[-1]*1.5)
        # self.sigmaList = [0.6, 1]
        # for i in range(10):
        #     self.sigmaList.append(self.sigmaList[-1]+1)
        # self.sigmaList.extend([20, 30])
        print(self.sigmaList)

        started = True
        for i in range(len(self.sigmaList)):
            sigma = self.sigmaList[i]
            g = scipy.ndimage.gaussian_laplace(self.gray, sigma=sigma)
            g *= sigma**2
            g = g**2
            self.showNormImage(g, sigma)
            if(started):
                self.scaleSpace = g
                started = False
            else:
                self.scaleSpace = np.dstack((self.scaleSpace, g))
    
    def nms(self):
        self.cx = []
        self.cy = []
        self.rad = []

        self.sigmaMatrix = np.zeros((self.h, self.w))
        neighborWidth = 1
        print("NMS...")
        for i in range(neighborWidth, self.h-neighborWidth):
            for j in range(neighborWidth, self.w-neighborWidth):
                ############# NMS ##############
                # for point scaleSpace[i][j][k], we check all its neighborhood value: 3x3xn
                # If the point is a extrema (largest, and substantially larger), then it's a interest point.
                for k in range(neighborWidth, len(self.sigmaList)-neighborWidth):
                    # neighbor = self.scaleSpace[i-neighborWidth:i+neighborWidth+1, j-neighborWidth:j+neighborWidth+1, k-neighborWidth:k+neighborWidth+1]
                    neighbor = self.scaleSpace[i-neighborWidth:i+neighborWidth+1, j-neighborWidth:j+neighborWidth+1, :]
                    if(self.scaleSpace[i][j][k] == neighbor.max()):
                        self.cx.append(i)
                        self.cy.append(j)
                        maxSigma = self.sigmaList[k]
                        self.rad.append(maxSigma*1.414)
                        self.sigmaMatrix[i][j] = maxSigma
        self.cx = np.array(self.cx)
        self.cy = np.array(self.cy)
        self.rad = np.array(self.rad)

        print("Now drawing circles...")
        self.show_all_circles(self.image, self.cy, self.cx, self.rad)
        # self.circleToEllipse()
        # self.show_ellipse()

    def circleToEllipse(self):
        self.ex = []
        self.ey = []
        self.er = []
        self.eo = []
        sobelx = cv2.Sobel(self.gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(self.gray,cv2.CV_64F,0,1,ksize=5)
        for x, y, r in zip(self.cx, self.cy, self.rad):
            if(x+r >= self.h or y+r >= self.w or x-r <0 or y-r<0):
                continue
            r = round(r)
            Ix = sobelx[x-r : x+r+1, y-r: y+r+1]
            Iy = sobely[x-r : x+r+1, y-r: y+r+1]
            size = Ix.shape[0]
            gk = np.zeros((size, size))
            gk[size//2][size//2] = 1
            # gk = cv2.GaussianBlur(gk, ksize=(size, size), sigmaX=Ix.shape[0]/6, sigmaY=Ix.shape[0]/6)
            # gk = cv2.resize(gk, Ix.shape)
            # print(gk)
            M = np.zeros((2, 2))
            M[0][0] = np.sum(Ix**2)
            M[0][1] = np.sum(Ix*Iy)
            M[1][0] = M[0][1]
            M[1][1] = np.sum(Iy**2)
            M = sp.Matrix(M)
            R, D = M.diagonalize()  
            R = np.array(R)
            D = np.array(D)
            lambda1 = D[0][0]
            lambda2 = D[1][1]
            scale = 2*r / (lambda1 + lambda2)
            lambda1 *= scale
            lambda2 *= scale
            ratio = lambda1/lambda2
            if(ratio < 0.1 or ratio > 10):
                continue
            self.ex.append(x)
            self.ey.append(y)
            self.er.append((lambda1, lambda2))
            self.eo.append(math.atan2(R[0][0], R[0][1]))
            # print(r, lambda1, lambda2)


                

    def showNormImage(self, image, sigma):
        norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        # print(normG)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image %3f' % (sigma,), norm)
        cv2.waitKey(0)

    def show_ellipse(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(image, cmap='gray')
        for x, y, r, angle in zip(self.ey, self.ex, self.er, self.eo):
            angle = angle/math.pi*180
            # print(x, y, r, angle)
            ellp = Ellipse((x, y), width=r[0]*2, height=r[1]*2, angle=angle, fill=False, color="red")
            ax.add_patch(ellp)

        plt.title('%i circles' % len(self.ey))
        plt.show()
        
    def show_all_circles(self, image, cx, cy, rad, color='r'):
        """
        image: numpy array, representing the grayscsale image
        cx, cy: numpy arrays or lists, centers of the detected blobs
        rad: numpy array or list, radius of the detected blobs
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
        for x, y, r in zip(cx, cy, rad):
            circ = Circle((x, y), r, color=color, fill=False, linewidth=1.5)
            ax.add_patch(circ)

        plt.title('%i circles' % len(cx))
        plt.show()


if __name__ == '__main__':
    image = cv2.imread("./data/butterfly.jpg")
    blobDetector = BlobDetector(image)
    blobDetector.buildScaleSpace()
    # blobDetector.buildScaleSpaceDownsample()
    # blobDetector.buildScaleSpaceGaussian()
    blobDetector.nms()
