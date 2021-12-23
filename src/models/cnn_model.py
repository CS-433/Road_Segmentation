import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation,LayerNormalization


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os,sys
import io

import sys 

import utilities.helpers_train
import utilities.helpers_run 
from utilities.helpers_train import *
from utilities.helpers_run import *

class Cnn(keras.Model):

    def __init__(self):
        super(Cnn, self).__init__()
        
        keras.backend.set_image_data_format('channels_last')

        self.patch_size = 16
        self.window_size = 80 #chosen such that enough context surround a patch of size 16
        self.nb_channels = 3 
        self.nb_classes = 2  
        self.batch_size = 256
        self.model()
        

    def layer(self,depth,filter):
        self.model.add(layers.Conv2D(depth, filter, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.MaxPool2D(padding='same'))
        self.model.add(layers.Dropout(0.2))
    
    def get_f1(self,y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
  
    def model(self):
       
        self.model = keras.Sequential()

        # Input layer
        input_shape = (self.window_size, self.window_size, self.nb_channels)
        self.model.add(layers.InputLayer(input_shape))

        #First layer 
        self.layer(64,5)
        

        #Second layer 
        self.layer(128,3)

        # Third layer 
        self.layer(256,3)
        self.model.add(layers.Flatten())

        # Fourth fully connected layer 
        self.model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-6)))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        self.model.add(layers.Dropout(0.5))

        # Softmax activation function
        self.model.add(
            layers.Dense(self.nb_classes, kernel_regularizer=keras.regularizers.l2(1e-6),
                         activation='softmax'))

        # Adam optimizer
        optimizer = keras.optimizers.Adam()

        # Binary cross_entropy loss
        self.model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                           metrics=[self.get_f1])
        
        self.model.summary()

    def train_model(self, gt_imgs, imgs, nb_epochs):
        """Trains the CNN model """

        padding_size = (self.window_size - self.patch_size) // 2 

        #Padding of the image
        imgs = np.asarray([pad(imgs[i], padding_size) for i in range(imgs.shape[0])])
        gt_imgs = np.asarray([pad(gt_imgs[i], padding_size) for i in range(imgs.shape[0])])
        
        
        history = self.model.fit_generator(batches_generator(imgs, gt_imgs, self.batch_size,                                    self.window_size,self.patch_size,self.nb_classes),
                                               steps_per_epoch=30,
                                               epochs=nb_epochs,
                                               verbose=1)
        
        
        return history
      
    

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
          labels = self.model.predict(x)
          return labels
          