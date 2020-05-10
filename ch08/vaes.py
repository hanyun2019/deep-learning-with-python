################################################################ 
# Haowen modified on May 9, 2020
#
# Deep Learning with Python
# Chapter 8: 
# 
# Chapter 8.4: Generating images with variational autoencoders
# 
################################################################

import keras
from keras import layers
from keras import backend as K 
from keras.models import Model
import numpy as np

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

# z = layers.Lambda(sampling)([z_mean, z_log_var])

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
# https://keras.io/examples/variational_autoencoder/
# z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
