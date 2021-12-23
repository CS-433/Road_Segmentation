
"""
Some helper functions for the training of the different models
"""



# Helpful import 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.backend as K
import re 
from tensorflow.keras.utils import to_categorical
import os,sys
import io

# Constants 
FOREGROUND_THRESHOLD=0.25



# F1 score
# source : https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d#:~:text=By%20default%2C%20f1%20score%20is,like%20accuracy%2C%20categorical%20accuracy%20etc.
def get_f1(y_true, y_pred): 
    """
    Calculates the f1 score for one epoch
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def display(history):
    """
    Displays the binary cross-entropy loss and f1 score for a training and validation set
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    get_f1= history.history['get_f1']
    val_get_f1 = history.history['val_get_f1']
    plt.plot(epochs,get_f1, 'y', label='get_f1')
    plt.plot(epochs, val_get_f1, 'r', label='val_get_f1')
    plt.title('Training and validation f1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.show()


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


def load_image(infilename):
    """
    loads an image
    """
    data = mpimg.imread(infilename)
    return data 


def pad(img, padding):
    """
    Adds padding to an image with a reflecting border
    """
    if len(img.shape) < 3:
        img = np.lib.pad(img, ((padding, padding), (padding, padding)), 'reflect')
    else:
        img = np.lib.pad(img, ((padding, padding), (padding, padding),(0,0)), 'reflect')
    return img


def patch_to_label(patch):
    """
    Maps a BW white patch image to a label using thresholding 
    """
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0



def batches_generator (imgs, gt_imgs, batch_size, window_size,patch_size,nb_classes):
  """Returns: input batch: a vector of size batch_size containing windows of size window_size * window _size * 3
     y_batch : vector of size batch_size containing vectors of size nb_classes.  
  """
  while True:
        input_batch = np.empty((batch_size, window_size, window_size, 3))
        y_batch = np.empty((batch_size, 2))

        for i in range(batch_size):
                    
                    window, gt_patch = create_window(imgs,gt_imgs, window_size,patch_size )
                    input_batch[i] = window
                    #outputs a binary vector: 1 in the cell of one of a class means that the gt_patch has been assigned to the                         corresponding class 
                    y_batch[i] = to_categorical(patch_to_label(gt_patch), nb_classes) 
        yield input_batch, y_batch
           


def create_window (img,gt_img, window_size,patch_size ):
    """
    Creates a window of size=patch_size for the image and groundtruth
    """
    #Pick a random image
    nb = np.random.choice(img.shape[0]) 
    img = img[nb]
    gt_img = gt_img[nb]
    #Dimensions of the window and patch
    min_ = window_size // 2
    max_ = img.shape[0]-min_
    lim = patch_size//2
    #Center of the windows and gt_patch
    center = np.random.randint(min_, max_, 2)
    #extraction of a window
    window = img[center[0] - min_:center[0] + min_,center[1] - min_:center[1] +min_]
    #extraction of the corresponding patch
    gt_patch = gt_img[center[0] - lim:center[0] + lim,center[1] -lim:center[1] + lim]
    return window, gt_patch





