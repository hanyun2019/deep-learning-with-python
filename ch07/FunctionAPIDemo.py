################################################### 
# Haowen modified on May 2, 2020
#
# Deep Learning with Python
# Chapter 7: Advanced deep-learning best practices
# 
# This chapter covers:
# The Keras functional API
# Using Keras callbacks
# Working with the TensorBoard visualization tool
# Important best practices for developing state-of-the-art models
# 
################################################### 

from keras.models import Sequential, Model
from keras import layers
from keras import Input

import numpy as np 

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()

# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 64)                0         
# _________________________________________________________________
# dense_4 (Dense)              (None, 32)                2080      
# _________________________________________________________________
# dense_5 (Dense)              (None, 32)                1056      
# _________________________________________________________________
# dense_6 (Dense)              (None, 10)                330       
# =================================================================
# Total params: 3,466
# Trainable params: 3,466
# Non-trainable params: 0
# _________________________________________________________________

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)

