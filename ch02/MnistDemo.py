# Haowen modified on Jan 1, 2020
#
# Deep Learning with Python
# Chapter 2: Before we begin: the mathematical building blocks of neural networks 
# The Mnist dataset
#

# import keras
from tensorflow import keras          # TensorFlow 2.0         
keras.__version__

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape: ",train_images.shape)
print("train_labels: ",train_labels)
print("test_images.shape: ",test_images.shape)
print("len(test_labels)): ",len(test_labels))
print("test_labels: ",test_labels)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)





