# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os,sys
import io
from PIL import Image
from IPython.display import clear_output
from patchify import patchify

#root_dir =  "/content/drive/MyDrive/MLprojet/training/"
"""
This file contains the functions necessary for the pre-processing steps on the training set (after data augmentation if needed)
"""


# FUNCTIONS

def load_image(infilename):
    """
    load 1 image
    """
    data = mpimg.imread(infilename)
    return data

def load_train(root_dir):
    """
    load all images of the training set
    """
    # Satellite images
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = len(files)
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    # Checking the dimension and format of the images
    print('The size of the masks is:', imgs[1].shape)
    print('The values are in the range [', imgs[1].min(), imgs[1].max(), ']')

    # Ground truth images
    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gts = [load_image(gt_dir + files[i]) for i in range(n)]
    # Checking the dimension and format of the images
    print('The size of the masks is:', gts[1].shape)
    print('The values are in the range [', gts[1].min(), gts[1].max(), ']')
    
    return imgs, gts

def load_test(root_dir):
    """
    load all images of the training set
    """
    n=50
    image_dir = root_dir + "images/"
    image_filenames_test = [image_dir + 'test_' + str(i + 1)+'.png' for i in range(n)]
    imgs = [load_image(image_filenames_test[i]) for i in range(n)]

    return imgs

def patch_list(imgs,masks,patch_size):
    """
    divide images in patches of size 'patch_size'
    return 2 arrays : 1 for satellite images, 1 for ground truth images
    """
    n = len(imgs)

    imgs = np.asarray(imgs)
    masks = np.asarray(masks)
    
    if(patch_size == 256):
        step_size =  144

    elif(patch_size == 128):
      step_size =  90
      
      # Reshape
      imgs = imgs[:, 1:399, 1:399,:]
      masks = masks[:, 1:399, 1:399]
    
    # Transform lists of images into array of images to facilitate processing
    img_patches = np.asarray([patchify(imgs[i], (patch_size, patch_size,3), step_size) for i in range(n)])
    gt_patches = np.asarray([patchify(masks[i], (patch_size, patch_size),step_size) for i in range(n)])
    
    img_patches = np.squeeze(img_patches,axis = 3)
    gt_patches = np.expand_dims(gt_patches, axis = 5)
    
    img_patches_list = []
    gt_patches_list = []
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            for k in range(img_patches.shape[1]):
                img_patches_list.append(img_patches[i,j,k,:,:])
                gt_patches_list.append( gt_patches[i,j,k,:,:])
    
    img_patches_list = np.asarray(img_patches_list)
    gt_patches_list = np.asarray(gt_patches_list)
    
    return img_patches_list, gt_patches_list




def create_patches(imgs, patch_size):
    n = len(imgs)
    if(patch_size == 256):
        step_size = 176
    elif(patch_size == 128):
      step_size = 120
    img_patches = np.asarray([patchify(imgs[i], (patch_size, patch_size,3),step_size) for i in range(n)])
    imgs = np.asarray(imgs)
    img_patches = np.squeeze(img_patches,axis = 3)

    return img_patches