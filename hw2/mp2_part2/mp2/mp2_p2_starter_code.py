# Libraries you will find useful
import numpy as np
import scipy 
import skimage
import matplotlib.pyplot as plt


# Starter code for Assignment 2 Part 2: Scale-space blob detection



# convert images to grayscale
# rescale the intensities to between 0 and 1 (simply divide them by 255 should do the trick)



# Create the Laplacian filter
# Pay careful attention to setting the right filter mask size. Hint: Should the filter width be odd or even?
from scipy.ndimage.filters import gaussian_laplace



# filtering the image (two implementations)
# one that increases filter size, and one that downsamples the image
# For timing, use time.time()



# nonmaximum suppression in scale space
# you may find functions scipy.ndimage.filters.rank_filter or scipy.ndimage.filters.generic_filter useful



# To display the detected regions as circle
from matplotlib.patches import Circle
def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()

