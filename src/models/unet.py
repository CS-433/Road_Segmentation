from tensorflow import keras
from tensorflow.keras import layers
from keras.models import *
from keras.layers import *

#Unet model

def encode(inputs,filter_n, max_pooling=True):
    filter= 32
    c1= tf.keras.layers.Conv2D(filters = filter*filter_n, kernel_size = 3, kernel_initializer = 'he_normal' , padding = 'same') (inputs)
    b1= tf.keras.layers.BatchNormalization()(c1)
    l1 = keras.layers.LeakyReLU(0)(b1)
    c2= tf.keras.layers.Conv2D(filters = filter*filter_n, kernel_size = 3, kernel_initializer = 'he_normal' , padding = 'same') (l1)
    b2= tf.keras.layers.BatchNormalization()(c2)
    l2 = keras.layers.LeakyReLU(0)(b2)
    if max_pooling :
        m1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(l2)
    else :
        m1=0
    return l2, m1


def decode(inputs_1, inputs_2, filter_n): 
    filter = 32
    ct1 = tf.keras.layers.Conv2DTranspose(filters= filter*filter_n , kernel_size = 3,strides = (2,2), padding = 'same')(inputs_1)
    b1= tf.keras.layers.BatchNormalization()(ct1)
    ccat1= tf.keras.layers.concatenate([b1, inputs_2])
    c1= tf.keras.layers.Conv2D(filters = filter*filter_n, kernel_size = 3, kernel_initializer = 'he_normal' , padding = 'same') (ccat1)
    b2= tf.keras.layers.BatchNormalization()(c1)
    l1 = keras.layers.LeakyReLU(0)(b2)
    c2= tf.keras.layers.Conv2D(filters = filter*filter_n, kernel_size = 3, kernel_initializer = 'he_normal' , padding = 'same') (l1)
    b3= tf.keras.layers.BatchNormalization()(c2)
    l2 = keras.layers.LeakyReLU(0)(b3)

    return l2

def unet_model(input_size): #input_size = (128,128,3)
    inputs = Input(input_size)
    
    #encode
    b1, m1 = encode(inputs, 1,  max_pooling = True)
    b2, m2 = encode(m1, 2,  max_pooling = True)
    b3, m3 = encode(m2, 4,  max_pooling = True)
    b4, m4 = encode(m3, 8, max_pooling = True)
    b5, m5 = encode(m4,16, max_pooling = False)

    #decode
    d1= decode(b5,b4, 8)
    d2 = decode(d1,b3, 4)
    d3 = decode(d2,b2, 2)
    d4 = decode(d3,b1, 1)

    #output
    output_layer= tf.keras.layers.Conv2D(filters=1, kernel_size =1, activation = 'sigmoid')(d4)
    
    model = Model(inputs = [inputs], outputs = [output_layer])
    model.summary()
    return model


