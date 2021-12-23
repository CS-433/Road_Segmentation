# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import *
from keras.layers import *
import dask.array as da
import keras.backend as K
import os
from sklearn.model_selection import train_test_split
import sys 

#  Install segmentation models
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

# Install patchify
from patchify import patchify


# Import from utilities
import utilities.helpers_train
import utilities.data_preprocessing
from utilities.helpers_train import *
from utilities.data_preprocessing import *
# Import from models
import models.unet 
import models.cnn_model
from models.unet import *
from models.cnn_model import *

#Variables
PATCHES = 256
TRAINING = 1 # 1: for training without validation set, 0: for training with validation set
TEST_SIZE = 0.2
MODEL = "UNET"
EPOCHS = 3
BATCH_SIZE = 16

root_dir = "../MLprojet/data/training/"
#root_dir = "../MLprojet/data/augmented_training/"    # CHANGER LE PATH EN FONCTION DATA AUG TESTEE


#Load a set of images from the root directory
new_imgs, new_gts  = load_train(root_dir)


if (MODEL =='CNN'):

    filepath = "../MLprojet/saved_models/model_name.h5"
    cnn = Cnn()
    new_gts = np.asarray(new_gts)
    new_imgs=  np.asarray(new_imgs)
    history = cnn.train_model(new_gts, new_imgs,EPOCHS)
    cnn.save_weights(filepath)

    print("model saved !")
else: 
    # Create Patches
    img_patches_list , gt_patches_list =  patch_list(new_imgs, new_gts, PATCHES)

    # delete unused data
    del new_imgs
    del new_gts

    # Split data 
    if(TRAINING == 1):
        X_train = img_patches_list
        y_train = gt_patches_list

        # Create chunk data
        X_tr = da.from_array(np.asarray(X_train), chunks=(1000, 1000, 1000, 1000))
        y_tr = da.from_array(np.asarray(y_train), chunks=(1000, 1000, 1000, 1000))
    elif(TRAINING == 0):
        X_train, X_test, y_train, y_test = train_test_split(img_patches_list, gt_patches_list, test_size = TEST_SIZE, random_state = 42)

        # Create chunk data
        X_tr = da.from_array(np.asarray(X_train), chunks=(1000, 1000, 1000, 1000))
        y_tr = da.from_array(np.asarray(y_train), chunks=(1000, 1000, 1000, 1000))
        X_te = da.from_array(np.asarray(X_test), chunks=(1000, 1000, 1000, 1000))
        y_te = da.from_array(np.asarray(y_test), chunks=(1000, 1000, 1000, 1000))

    if (MODEL == "UNET"):
        input_size = (PATCHES,PATCHES,3)
        model = unet_model(input_size)

    elif(MODEL == "RESNET34"):
        #Model from segmentation models
        BACKBONE1 = 'resnet34'
        preprocess_input = sm.get_preprocessing(BACKBONE1)

        X_tr = preprocess_input(X_tr)
        if(TRAINING == 0):
            X_te1 = preprocess_input(X_te)

        model = sm.Unet(BACKBONE1, classes=1, activation='sigmoid')

    elif(MODEL == "SERESNET34"):
        #Model from segmentation models
        BACKBONE2 = 'seresnet34'
        preprocess_input = sm.get_preprocessing(BACKBONE2)

        X_tr = preprocess_input(X_tr)
        if(TRAINING == 0):
            X_te1 = preprocess_input(X_te)

        model = sm.Unet( BACKBONE2, classes=1, activation='sigmoid')


    elif(MODEL == "RESNET50"):
        #Model from segmentation models
        BACKBONE3 = 'resnet50'
        preprocess_input = sm.get_preprocessing(BACKBONE3)

        X_tr = preprocess_input(X_tr)
        if(TRAINING == 0):
            X_te1 = preprocess_input(X_te)

        model = sm.Unet( BACKBONE3, classes=1, activation='sigmoid')


    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss = 'binary_crossentropy', metrics=[get_f1])

    if(TRAINING == 1):
        #train model without validation set
        history = model.fit(X_tr, y_tr, batch_size = BATCH_SIZE, verbose=1, epochs=EPOCHS)  

    elif(TRAINING == 0):
        #train model with validation set
        history = model.fit(X_tr, y_tr, batch_size = BATCH_SIZE, verbose=1, epochs=EPOCHS, validation_data=(X_te, y_te))  

        #plot the training and validation loss and f1 score at each epoch
        display(history)

    #Save the model for future use
    #model.save("../MLprojet/saved_models/model_name.hdf5")
    #model.save("../MLprojet/saved_models/model_name.hdf5")
    #print("model saved !")
