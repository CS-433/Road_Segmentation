# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import imutils
import random
import scipy

%load_ext autoreload
%autoreload 2


"""
This file perform data augmentation of the training set of satellite-ground truth images.
Need to been run only once with the desired augmentations, the resulting images are then stored in a directory in he 'training' directory
"""


# FUNCTIONS

def load_image(infilename):
    """
    load 1 image
    """
    data = mpimg.imread(infilename)
    return data


# Functions for data augmentation


def augment_one(img, mask, n):
    """
    performs transformations for the augmentation of 1 image
    """
    
    # Find the repository to save the augmented data :
    save_dir = "training/augmented_4rot180_Gblur5_Flips/"
    save_dirimg = save_dir + "images/"
    save_dirmask = save_dir + "groundtruth/"
    
    # Save original image and mask
    nb = n*10+1
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', mask)#, cmap='gray')
    nb += 1
    
    LEN_IMG = 400
    
    # Data Augmentation and saving of the new data
    # Horizontal flip
    hflip_img = np.flip(img, 1)
    hflip_mask = np.flip(mask, 1)
    
    # Vertical flip on the original images
    vflip_img = np.flip(img, 0)
    vflip_mask = np.flip(mask, 0)
    
    # Vertical flip on the horizontally flipped images
    vhflip_img = np.flip(hflip_img, 0)
    vhflip_mask = np.flip(hflip_mask, 0)
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', vhflip_img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', vhflip_mask)
    nb += 1
    
    # A 45° rotation on original images
    rot45_img = scipy.ndimage.interpolation.rotate(img, 45, reshape=False, mode="reflect")
    rot45_mask = scipy.ndimage.interpolation.rotate(mask, 45, reshape=False, mode="reflect")
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', rot45_img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', rot45_mask)
    nb += 1
    
    # 4 random rotations of angles between 0 and 180°   
    for i in range(4):
        angle = random.randrange(0,180)
        print(angle)
        rot_hvflip_img = scipy.ndimage.interpolation.rotate(img, angle, reshape=False, mode="reflect")
        rot_hvflip_mask = scipy.ndimage.interpolation.rotate(mask, angle, reshape=False, mode="reflect")
        cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', rot_hvflip_img)
        cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', rot_hvflip_mask)
        nb += 1

    # Gaussian blur
    blur_img = cv2.GaussianBlur(img, (5,5), 0)
    blur_mask = cv2.GaussianBlur(mask, (5,5), 0)
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', blur_img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', blur_mask)
    nb += 1   
    
    # Lighting effects : surexposed images
    light_img = cv2.convertScaleAbs(img, alpha=1.3, beta=40) 
    light_mask = cv2.convertScaleAbs(mask, alpha=1.3, beta=40)
    print('Lighting', light_img.shape)
    print('Lighting', light_mask.shape)
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', light_img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', light_mask)
    nb += 1
    
    # Lighting effect : underexposed images
    dark_img = cv2.convertScaleAbs(img, alpha=0.7, beta=0) 
    dark_mask = cv2.convertScaleAbs(mask, alpha=0.7, beta=0)
    cv2.imwrite(save_dirimg + 'satImage_' + str(nb).zfill(4) + '.png', dark_img)
    cv2.imwrite(save_dirmask + 'satImage_' + str(nb).zfill(4) + '.png', dark_mask)
    nb += 1
    
    return


def augment_all(n, imgs, masks):
    """
    call the function 'augment_one()' for each set of satellite-ground truth images
    to perform data augmentation of the entire training set
    """
    
    for i in range(n):
        print(i)
        augment_one(imgs[i], gt_imgs[i], i)
        print(aug_img.shape)
    
    return

# ----------------------------------------Main()


# Load the original set of training images (pairs satellite-ground truth images)

# Satellite images
root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
# Checking the dimension and format of the images
print('The size of the images is:', imgs[1].shape)
print('The values are in the range [', imgs[1].min(), imgs[1].max(), ']')

# Ground truth images
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
# Checking the dimension and format of the images
print('The size of the masks is:', gt_imgs[1].shape)
print('The values are in the range [', gt_imgs[1].min(), gt_imgs[1].max(), ']')


# Data Augmentation

augment_all(n, imgs, gt_imgs)