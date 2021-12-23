# Useful Imports
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import *
from keras.layers import *
from keras.models import load_model

# Import segmentation models
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

# Import patchify
from patchify import patchify, unpatchify


# Import helpers
import sys 
import utilities.helpers_run
import utilities.data_preprocessing
from utilities.helpers_run import *
from utilities.data_preprocessing import *
import models.cnn_model 
from models.cnn_model  import *

# Variables
PATCHES = 256
MODEL_NR = 1
CNN = False

#Load test images
root_dir = "../MLprojet/data/test_set/"


if CNN:
    #Load test images
    
    filepath = "../MLprojet/saved_models/cnn_aug.h5"
    image_dir = root_dir + "images/"
    image_filenames_test = [image_dir + 'test_' + str(i + 1)+'.png' for i in range(50)]
    cnn = Cnn()
    cnn.built = True
    cnn.load_weights(filepath)
    submission_path = '../MLprojet/submission_cnn.csv'
    masks_to_submission_cnn(cnn, submission_path, *image_filenames_test,window_size=80,patch_size=16)

else:
    imgs = load_test(root_dir)

    # Create Patches
    img_patches = create_patches(imgs,PATCHES)

    #Load model(s)
    if(MODEL_NR == 1):
        model = load_model("../MLprojet/saved_models/resnet50_aug.hdf5", compile=False)
    elif(MODEL_NR == 3):
        model1 = load_model("../MLprojet/saved_models/resnet34_aug.hdf5", compile=False)
        model2 = load_model("../MLprojet/saved_models/seresnet34_aug.hdf5", compile=False)
        model3 = load_model("../MLprojet/saved_models/resnet50_aug.hdf5", compile=False) 

        #Preprocess test data
        BACKBONE1 = 'resnet34'
        preprocess_input1 = sm.get_preprocessing(BACKBONE1)
        img_patches1 = preprocess_input1(img_patches)

        BACKBONE2 = 'seresnet34'
        preprocess_input2 = sm.get_preprocessing(BACKBONE2)
        img_patches2 = preprocess_input2(img_patches)

        BACKBONE3 = 'resnet50'
        preprocess_input3 = sm.get_preprocessing(BACKBONE3)
        img_patches3 = preprocess_input3(img_patches)

    #Predict on model(s)
    if(MODEL_NR == 1):
        img_predict_patch = prediction(img_patches, model)
    elif(MODEL_NR == 3):
        img_predict_patch = multi_prediciton(img_patches1, img_patches2, img_patches3, model1, model2, model3)
        img_predict_patch = np.array(img_predict_patch)
        img_predict_patch = np.squeeze(img_predict_patch, 1)


    # Reshaping prediction
    img_predict_patch = np.reshape(img_predict_patch,(img_patches.shape[0],img_patches.shape[1],img_patches.shape[2], PATCHES,PATCHES) )
    img_predict_patch.shape

    # Sanity check
    sanity_check(PATCHES,img_predict_patch)

    ## Reconstruct full image
    reconstructed_img  = np.asarray([unpatchify(img_predict_patch[i],(608,608)) for i in range(len(imgs))])
    print(reconstructed_img.shape)

    # Create predicition path
    prediction_path="../MLprojet/prediction_unet/"
    for i in range(reconstructed_img.shape[0]):
        test1 = np.squeeze(reconstructed_img[i]).round()
        test = img_float_to_uint8(test1)
        prediction_name = prediction_path + 'pred_' + str(i + 1) + '_unet.png'
        Image.fromarray(test).save(prediction_name)

    image_filenames_predict = [prediction_path + 'pred_' + str(i + 1) + '_unet.png' for i in range(reconstructed_img.shape[0])]
    

    # Create submission
    submission_filename= '../MLprojet/submissions/submission_multinet_aug.csv'
    masks_to_submission(submission_filename, *image_filenames_predict)



