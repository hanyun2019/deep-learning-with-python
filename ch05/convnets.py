################################################ 
# Haowen modified on Jan 5, 2020
#
# Deep Learning with Python
# Chapter 5: Deep learning for computer vision
# 5.1-introduction-to-convnets.ipynb
# The Mnist dataset
################################################ 

import keras
# from tensorflow import keras    # TensorFlow 2.0   
keras.__version__

# We will use our convnet to classify MNIST digits, a task that you've already been through in Chapter 2, using a densely-connected network (our test accuracy then was 97.8%). 
# Even though our convnet will be very basic, its accuracy will still blow out of the water that of the densely-connected model from Chapter 2.

# The 6 lines of code below show you what a basic convnet looks like. It's a stack of Conv2D and MaxPooling2D layers. 
# Importantly, a convnet takes as input tensors of shape (image_height, image_width, image_channels) (not including the batch dimension). 
# In our case, we will configure our convnet to process inputs of size (28, 28, 1), which is the format of MNIST images. We do this via passing the argument input_shape=(28, 28, 1) to our first layer.

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Let's display the architecture of our convnet so far:

model.summary()

# Output:
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
# =================================================================
# Total params: 55,744
# Trainable params: 55,744
# Non-trainable params: 0

# You can see above that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). 
# The width and height dimensions tend to shrink as we go deeper in the network. 
# The number of channels is controlled by the first argument passed to the Conv2D layers (e.g. 32 or 64).
# 
# The next step would be to feed our last output tensor (of shape (3, 3, 64)) into a densely-connected classifier network like those you are already familiar with: a stack of Dense layers. 
# These classifiers process vectors, which are 1D, whereas our current output is a 3D tensor. 
# So first, we will have to flatten our 3D outputs to 1D, and then add a few Dense layers on top:

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Output:
# _________________________________________________________________
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 576)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                36928     
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0

# As you can see, our (3, 3, 64) outputs were flattened into vectors of shape (576,), before going through two Dense layers.
# Now, let's train our convnet on the MNIST digits. We will reuse a lot of the code we have already covered in the MNIST example from Chapter 2.

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Let's evaluate the model on the test data:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test_acc: ",test_acc)

# While our densely-connected network from Chapter 2 had a test accuracy of 97.8%, 
# our basic convnet has a test accuracy of 99.3%: we decreased our error rate by 68% (relative). Not bad!
