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
# Core
import sys
import math
import numpy as np
from ctypes import c_uint8

# Plot
import matplotlib.pyplot as plt

# Depth
import robosuite.utils.camera_utils as camera_utils

# Render
import pygame

# Image processing
import cv2

def render_images(object,env_obs,second_image):
    if object.use_pygame_render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if object.visualize_camera_obs:
            # Display the camera observation
            if not object.use_depth_obs:
                im = env_obs['image_'+object.camera_names[0]]
                im = np.uint8(im * 255.0)

            else:
                im1 = env_obs['image_'+object.camera_names[0]][:,:,0]
                im2 = env_obs['image_'+object.camera_names[0]][:,:,1]
                im = np.hstack((im1,im2))
                im = np.uint8(im * 255.0)

            im = cv2.merge([im,im,im])
        
        else:
            # Display agentview camera
            im = object.sim.render(camera_name="agentview", height=300, width=300)[::-1]
            im = np.flip(im.transpose((1, 0, 2)), 0)[::-1]

            # Display a 2nd image as part of a smaller inslet on bottom left
            if not object.use_depth_obs and not object.use_gray_img:
                inset1 = second_image # Display segmented eye-in-hand as an inset of the original image
                inset1 = np.uint8(inset1 * 255.0)
                inset1 = cv2.merge([inset1,inset1,inset1])
                inset1 = np.flip(inset1.transpose((1, 0, 2)), 0)[::-1] #for obs image visualization in pygame
                inset1 = cv2.resize(inset1, (80,80))

                im[:np.shape(inset1)[0],-np.shape(inset1)[1]:,:] = inset1

            elif not object.use_depth_obs and object.use_gray_img:

                # Display on bottom left: segmented image
                inset1 = object._observables[object.camera_names[0]+'_segmentation_instance'].obs
                inset1 = np.uint8(inset1 * (255.0/np.max(inset1)) )
                inset1 = cv2.merge([inset1,inset1,inset1])
                inset1 = cv2.resize(inset1, (80,80))
                im[:np.shape(inset1)[0],-np.shape(inset1)[1]:,:] = inset1

                # Display on bottom right: gray masked image
                inset2 = second_image
                inset2 = cv2.merge([inset2,inset2,inset2])
                # inset2 = np.flip(inset2.transpose((1, 0, 2)), 0)[::-1] #for obs image visualization in pygame
                inset2 = cv2.resize(inset2, (80,80))
                im[-np.shape(inset2)[0]:,-np.shape(inset2)[1]:,:] = inset2

            else:
                inset1 = env_obs['image_'+object.camera_names[0]][:,:,0]
                inset1 = np.uint8(inset1 * 255.0)
                inset1 = cv2.merge([inset1,inset1,inset1])
                inset1 = cv2.resize(inset1, (80,80))
                im[:np.shape(inset1)[0],-np.shape(inset1)[1]:,:] = inset1

                inset2 = env_obs['image_'+object.camera_names[0]][:,:,1]
                inset2 = np.uint8(inset2 * 255.0)
                inset2 = cv2.merge([inset2,inset2,inset2])
                inset2 = cv2.resize(inset2, (80,80))
                im[-np.shape(inset2)[0]:,-np.shape(inset2)[1]:,:] = inset2

        pygame.pixelcopy.array_to_surface(object.screen, im)
        pygame.display.update()  

def process_gray_mask(rgb_im, seg_im, output_size):
    """
    Helper function to produce a masked gray image, where the fingers remain segmented as white.
    
    rgb->gray->bitwise mult with mask
    """

    # 01a Convert rgb image to gray image

    # Different possible ways to encode gray images. Want a scheme that is lighter to show the robot's black fingers
    gray_img = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2GRAY)   # 01 LUMA std in cv
    # gray_img = np.empty_like(output_size) 
    # gray_img = np.mean(rgb_im,axis=2)                   # 02 Intensity: average of rgb --> still dark    

    # 01b. Adjust Contrast

    # Get Hist
    # hist = cv2.calcHist(gray_img,[0],None,[256],[0,256])
    # plt.plot(hist)
    # plt.show()
    
    # Using an equalized hist, adjust contrasts in gray_img 
    gray_img=cv2.equalizeHist(gray_img)
    # cv2.imshow('equalized img',gray_img)
    # cv2.destroyAllWindows()  

    # Display adjusted histogram
    # hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])
    # plt.plot(hist)
    # plt.show()

    # clahe (Contrast Limited Adaptive Histogram Equalization): improves contrast
    clahe = cv2.createCLAHE(clipLimit=40)
    gray_img = clahe.apply(gray_img)
    # cv2.imshow('clahe img',gray_img)
    # cv2.destroyAllWindows()    

    #----------
    # Next 2 algos --> binary images not gray imgs
    #----------
    # Adaptive Contrast --> BINARY IMAGES  
    # bloc_sz = 41
    # mean = 0
    # gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block_sz, mean)
    # cv2.imshow('adaptive thresh',gray_img)
    # cv2.destroyAllWindows()  

    # ret,gray_img = cv2.threshold(gray_img, 0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('otsu img', gray_img)
    # cv2.destroyAllWindows()  

    #------------    
    
    # 02 Mask
    # Prepare Mask

    # Deterministic shuffling of values to map each geom ID to a random int in [0, 255]
    rstate = np.random.RandomState(seed=10)
    inds = np.arange(256)
    rstate.shuffle(inds)

    # 2 images: one for object only and one for gripper only 

    # Gripper: only retain 1s for the gripper
    gripper_img = seg_im.squeeze().astype('uint8').copy()
    gripper_img[gripper_img==0]=0
    gripper_img[gripper_img==1]=0
    gripper_img[gripper_img==2]=0
    gripper_img[gripper_img==3]=0
    gripper_img[gripper_img==4]=0
    gripper_img[gripper_img==5]=1 

    # Use morphological operators to remove noise
    kernel = np.ones((2,2),np.uint8)
    # gripper_img = cv2.erode(gripper_img,kernel,iterations = 1)
    gripper_img = cv2.morphologyEx(gripper_img, cv2.MORPH_OPEN, kernel)
    
    # Object: only retain 1's for the object
    object_img = seg_im.squeeze().astype('uint8').copy()
    object_img[object_img==0]=0
    object_img[object_img==1]=0
    object_img[object_img==2]=1 
    object_img[object_img==3]=0
    object_img[object_img==4]=0
    object_img[object_img==5]=0 

    # 03 Do and element-wise multiplication between gray and object_image
    gray_mask_img = gray_img*object_img

    # 04 Overlay gripper values as 255 in the gray object image (fingers appear as white)
    np.putmask(gray_mask_img, gripper_img,c_uint8(-1).value) #ie. 255

    # Switch type to float 32
    image_float = np.ascontiguousarray(gray_mask_img, dtype=np.float32)

    return cv2.resize(image_float, output_size)

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
    depth_im = camera_utils.get_real_depth_map(depth_im) #(object.sim,depth_im)
    
    # depth_im = np.flip(depth_im.transpose((1, 0, 2)), 1).squeeze(-1).astype('float64')
    depth_im = depth_im.squeeze(-1).astype('float64')

    return cv2.resize(depth_im, output_size)

def compute_blob_orientation(img=None, plot_flag=0):
    '''
    This method wishes to compute the necessary z-world rotation needed by the gripper to properly grasp 
    an object. Namely, to place its fingers around the minor-axis of the segmented-blob of the object. 
    
    To do this, we must first:
    1. Get a segmented image
    2. Compute the x,y mean-centered coordinates
    3. Compute the covariance of the coordinates
    4. Compute the eigenvals/eigenvecs of the covariance
    
    A plot analysis, compared with the original segmented will show that the vector representing the minor axis
    is the exact rotation that we have for the gripper to place its fingers along the minor (shorter) edge of an object.
    
    Note that if the evec of the minor axis starts with a negative x-value, we can simply take the negative 
    of the evec, to always get a rotation along the 1st and 2nd quadrants. 
    
    We will return the minor eigenvector in the first two quadrants. 

    If there is no blob, we return a null vector

    params:
        img (ndarray): a gray image (1-channel) of segmented object (x,y) starting on top left
    
    returns:
        object_orientation (ndarray): 2D vector for minor axes. Vector always adjusted to be in quadrants 1/4: [-pi/2,pi/2]
    
    raises:
        NotImplementedError for no ndarray
    '''
    if type(img) is not np.ndarray:
        raise NotImplementedError('The input to compute_blob_orientation should be a gray image')

    # Verify there is a blob. Otherwise, if all values are zero return null vec
    if not np.any(img):  
        object_orientation = np.zeros(2)
    else:
        # We want the indexes of the white pixels to find the axes of the blob.
        y,x = np.nonzero(img) # returns idx pos [....x.....],[....y....]. Ie for an np.eye(3): [0,1,2][0,1,2]

        # 02 Subtract mean
        x = x- np.mean(x)
        y = y- np.mean(y)
        coords = np.vstack([x,y])

        # Also check here for potential [0,0] coordinates
        if np.any(coords):             

            # 03 Calc covariance mat
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov) # 3 evals --> 1323, 2942 # evecs--> (2,2)

            # 04 Sort eigenvals in INCREASING order (minor axis first)
            sort_indices = np.argsort(evals)  # np.argsort returns indices in increasing order. # [::-1] reverses the order.
            evecs = evecs[sort_indices]

            # Plot it
            if plot_flag:
                
                # Plot major-minor...Sort eigenvals in decreasing order
                sort_indices = np.argsort(evals)[::-1]  # np.argsort returns indices in increasing order. # [::-1] reverses the order.
                x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue: major-axis
                x_v2, y_v2 = evecs[:, sort_indices[1]]  # Eigenvector with smaller eigenvalue: minor-axis
                    
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

            # Check if minor-axis is pointing in the negative direction, if so, multiply (x,y) by -1
            if evecs[0,0] < 0: 
                evecs = -1*evecs

            # Return the minor axis [:,0]
            object_orientation = evecs[:,0].reshape(1,-1, order='F').ravel()
        else:
            object_orientation = np.zeros(2)
    
    return object_orientation

def rotation(image, angle,mode='cutCorner',displayImg=None):
    '''
    This method rotates images about the plane given angleInDegrees, where positive is counterclockwise.
    The cv2 method is significantly faster than the equivalent process with ndimage.rotate.

    Two modes for this method: 
    1) lossy: no resizing of the image, loss of information
    2) lossless: img sz is increased, but not loss of information. 

    params:
        image (ndarray or list of ndarray): 1 or n-channel image. operation only on first channel
        angle (float): rotation angle in degrees. 
        mode (str): 'lossly' or 'lossless'
        displayImg (bool): if want to see single img set to True

    returns:
        rotated (ndarray or list of ndarray)

    raises:

    '''

    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)
    rot_mat     = cv2.getRotationMatrix2D(img_c,angle,1.0)

    if mode == 'lossy':    
        new_image   = cv2.warpAffine(image, M=rot_mat, dsize=(w,h))

    else: # mode == 'lossless':
        rad = math.radians(angle)
        sin = math.sin(rad)
        cos = math.cos(rad)

        # sin cos --> new width & height of rotated image, can be used so that the image is not cut off.
        # use new data to adjust for translation
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot_mat[0, 2] += ((b_w / 2) - img_c[0])
        rot_mat[1, 2] += ((b_h / 2) - img_c[1])

        # outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        new_image = cv2.warpAffine(image, M=rot_mat, dsize=(b_w, b_h))

    if displayImg:
        cv2.imshow('Original image', image)
        cv2.imshow('Rotated image', new_image)
        cv2.waitKey(0) # wait indefinitely, press any key on keyboard to exit

    return new_image
