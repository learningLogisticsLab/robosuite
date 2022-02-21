#major_minor_axes.py
#-------------------------------------------------------------------------------
# Covariance matrix and eigens
#-------------------------------------------------------------------------------
# Covariance matrices:
# Cov mat are square and symmetric. Summarize variance between two vars (x,y)
# C=[variance(x,x)variance(x,y)
#    variance(x,y)variance(y,y)],
#
#-------------------------------------------------------------------------------
# PCA:
#-------------------------------------------------------------------------------
# Use PCA to extract eigenvecs and eigenvalues
# Eigenvec with largest eigenval of a cov matrix gives us the direction along which the data has the largest variance
#
#-------------------------------------------------------------------------------
# Overview:
#-------------------------------------------------------------------------------
# 1. Matrix of points: (2,-1)
#    - First row are x coords
#    - Second row are y coords
# 2. Subtract each (x,y) with mean --> centers the data and minimizes the MSE
# 3. Calculate the 2x2 cov matrix
# 4. Find the 2 eigne-pairs of the dataset
# 5. Rearrange the eign-pairs in decreasing eigen-value order 
# 6. plot the principal components
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import imageio
from spatialmath import SO2

import robosuite.utils.camera_utils as camera_utils
import matplotlib.cm as cm
import cv2

def process_seg_image(seg_im, output_size):
    """
    Helper function to visualize segmentations as grayscale frames.
    
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """

    # flip and ensure all values lie within [0, 255]
    # seg_im = np.mod(np.flip(seg_im.transpose((1, 0, 2)), 1).squeeze(-1)[::-1], 256)
    seg_im = np.mod(seg_im.squeeze(-1), 256)


    # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
    rstate = np.random.RandomState(seed=10)
    inds = np.arange(256)
    rstate.shuffle(inds)

    # use @inds to map each geom ID to a color
    # gray_image = (cm.gray(inds[seg_im], 3))[..., :1].squeeze(-1).astype('float64')

    seg_im[seg_im==0]=0
    seg_im[seg_im==1]=0
    seg_im[seg_im==2]=1 #object to be grasped
    seg_im[seg_im==3]=0
    seg_im[seg_im==4]=0
    seg_im[seg_im==5]=1 #gripper

    image_float = np.ascontiguousarray(seg_im, dtype=np.float32)

    return cv2.resize(image_float, output_size)

def process_seg_obj_image(seg_im, output_size):
    """
    Helper function to visualize segmentations as grayscale frames.
    
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """

    # flip and ensure all values lie within [0, 255]
    # seg_im = np.mod(np.flip(seg_im.transpose((1, 0, 2)), 1).squeeze(-1)[::-1], 256)
    seg_im = np.mod(seg_im.squeeze(-1), 256)


    # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
    rstate = np.random.RandomState(seed=10)
    inds = np.arange(256)
    rstate.shuffle(inds)

    # use @inds to map each geom ID to a color
    # gray_image = (cm.gray(inds[seg_im], 3))[..., :1].squeeze(-1).astype('float64')

    seg_im[seg_im==0]=0
    seg_im[seg_im==1]=0
    seg_im[seg_im==2]=1 #object to be grasped
    seg_im[seg_im==3]=0
    seg_im[seg_im==4]=0
    seg_im[seg_im==5]=0 #gripper

    image_float = np.ascontiguousarray(seg_im, dtype=np.float32)

    return cv2.resize(image_float, output_size)    

def process_depth_image(depth_im, output_size):
    """
    Process depth map. Unscale and flip.
    """
    depth_im = camera_utils.get_real_depth_map(self.sim, depth_im)
    
    # depth_im = np.flip(depth_im.transpose((1, 0, 2)), 1).squeeze(-1).astype('float64')
    depth_im = depth_im.squeeze(-1).astype('float64')

    return cv2.resize(depth_im, output_size)

def compute_blob_orientation(img=None, plot_flag=0):
    '''
    This method computes the major-minor axis of the segmented blob. 
    It computes the eigen vectors of the blob and arranages them in order of decreasing eigenvalue order
    The eigenvector with the largest eigenvector will indicate a vector along the major axes.

    If there is no blob, we return a null vector

    params:
        img (ndarray): a gray image (1-channel) of segmented object
    
    returns:
        object_orientation (ndarray): 2D vector for major axes
    
    raises:

    '''
    # Verify there is a blob. Otherwise, if all values are zero return null vec
    if not np.any(img):  
        object_orientation = np.zeros([1,2])
    else:
        # We want the indexes of the white pixels to find the axes of the blob.
        y,x = np.nonzero(img) # still not clear 

        # 02 Subtract mean
        x = x- np.mean(x)
        y = y- np.mean(y)
        coords = np.vstack([x,y])

        # Also check here for potential [0,0] coordinates
        if np.any(coords):             

            # 03 Calc covariance mat
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov) # 3 evals --> 1323, 2942 # evecs--> (2,2)

            # 04 Sort eigenvals in decreasing order
            sort_indices = np.argsort(evals)[::-1]  # np.argsort returns indices in increasing order. # [::-1] reverses the order.
            x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue: major-axis
            x_v2, y_v2 = evecs[:, sort_indices[1]]  # Eigenvector with smaller eigenvalue: minor-axis

            # Plot it
            if plot_flag:
                scale = 20

                # Draw major-axis: scale it up to make line longer. Also don't use plot vec, but the anti-vec to get the whole axis (multiply by -scalar)
                plt.plot([x_v1*-scale*2, x_v1*scale*2],
                        [y_v1*-scale*2, y_v1*scale*2], 
                        color='red')

                # Draw minor-axis        
                plt.plot([x_v2*-scale, x_v2*scale],
                        [y_v2*-scale, y_v2*scale], 
                        color='blue')

                # Plots original oval x,y points in black
                plt.plot(x, y, 'k.')

                plt.axis('equal')        # Fixes scaling between y and x
                plt.gca().invert_yaxis()  # Inverts the y-axis (positive bottom, negative top)
                plt.show()

            # # Calculate offset with vertical: simply use arctan(opposite/adjacent) 
            # #   Where, the adjacent corresponds to the x-axis and the opp corresponds to the y-axis-->arctan(x,y)
            # theta = np.arctan((y_v1)/(x_v1))
            # object_orientation = SO2(theta) # # Use inverse rotation matrix to align end-effector with

            # Orientation is equivalent to the eigen-vectors directly. 
            # Use order F to keep (xx,xy and yx, yy)
            # The first eigenvector is enough to indicate the main orientation
            object_orientation = evecs[:,0].reshape(1,-1, order='F')
        else:
            object_orientation = np.zeros([1,2])
    
    return object_orientation

