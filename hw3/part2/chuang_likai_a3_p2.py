# imports
import os
import sys
import glob
import re
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import random


#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])


#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    processed_imarray = []
    for i in range(imarray.shape[2]):
        im = imarray[:,:,i]
        im = im - ambimage
        im[im < 0] = 0
        # im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
        im = im / (np.max(im) - np.min(im))
        if(i==0):
            processed_imarray = im
        else:
            processed_imarray = np.dstack((processed_imarray, im))
    # processed_imarray = np.array(processed_imarray)
    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    print(light_dirs.shape)
    h, w = imarray.shape[0: 2]
    print(h, w)
    albedo_image = np.zeros((h, w))
    surface_normals = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            I = imarray[i, j, :]
            S = light_dirs
            S_pinv = np.linalg.pinv(S)
            N = np.matmul(S_pinv,  I)
            # N = np.linalg.lstsq(S, I, rcond=None)[0]
            mag = np.linalg.norm(N)
            surface_normals[i, j, :] = N/mag
            albedo_image[i][j] = mag
    return albedo_image, surface_normals



def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    h, w = surface_normals.shape[0: 2]
    height_map = np.zeros((h, w))
    if(integration_method == 'row'):
        # first row first.
        for i in range(h):
            for j in range(w):
                p, q, r = surface_normals[i, j, :] / surface_normals[i, j, 2]
                # print(p, q, r)
                if(i==0 and j==0):
                    height_map[i][j] = 0
                elif(i == 0):
                    # first row
                    height_map[i][j] = height_map[i][j-1] + p
                else:
                    # Add downwards
                    height_map[i][j] = height_map[i-1][j] + q
    elif(integration_method == 'column'):
        # first column first
        for j in range(w):
            for i in range(h):
                p, q, r = surface_normals[i, j, :] / surface_normals[i, j, 2]
                # print(p, q, r)
                if(i==0 and j==0):
                    height_map[i][j] = 0
                elif(j == 0):
                    # first column
                    height_map[i][j] = height_map[i-1][j] + q
                else:
                    # Add to the right
                    height_map[i][j] = height_map[i][j-1] + p
    elif(integration_method == 'average'):
        hmap1 = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                p, q, r = surface_normals[i, j, :] / surface_normals[i, j, 2]
                # print(p, q, r)
                if(i==0 and j==0):
                    hmap1[i][j] = 0
                elif(i == 0):
                    hmap1[i][j] = hmap1[i][j-1] + p
                else:
                    hmap1[i][j] = hmap1[i-1][j] + q
        hmap2 = np.zeros((h, w))
        for j in range(w):
            for i in range(h):
                p, q, r = surface_normals[i, j, :] / surface_normals[i, j, 2]
                # print(p, q, r)
                if(i==0 and j==0):
                    hmap2[i][j] = 0
                elif(j == 0):
                    hmap2[i][j] = hmap2[i-1][j] + q
                else:
                    hmap2[i][j] = hmap2[i][j-1] + p
        height_map = (hmap1 + hmap2) /2
    elif(integration_method == 'random'):
        n = 5
        for i in range(h):
            for j in range(w):
                if(i==0 and j==0):
                    height_map[i][j] = 0
                    continue
                # Experiment n times from (0,0) to (x, y)
                cumsum = 0
                for k in range(n):
                    # for each path, we need go down for i times, and right for j times.
                    # shuffle the order
                    goDown = np.zeros(i)
                    goRight = np.ones(j)
                    path = np.append(goRight, goDown)
                    random.shuffle(path)
                    x = 0
                    y = 0
                    for move in path:
                        if(move == 0):
                            # go Down
                            x += 1
                            p, q, r = surface_normals[x, y, :] / surface_normals[x, y, 2]
                            cumsum += q
                        elif(move == 1):
                            # go Right
                            y += 1
                            p, q, r = surface_normals[x, y, :] / surface_normals[x, y, 2]
                            cumsum += p
                height_map[i][j] = cumsum / n

    return height_map



# Main function
if __name__ == '__main__':
    root_path = './croppedyale/croppedyale/'
    subject_name = 'yaleB01'
    integration_method = 'random'
    save_flag = True

    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name, 64)
    # print(imarray.shape)
    # plt.figure()
    # plt.imshow(ambient_image)
    # plt.show()
    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                    light_dirs)

    height_map = get_surface(surface_normals, integration_method)

    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)

    plt.show()




