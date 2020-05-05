################################################ 
# Haowen modified on Jan 4, 2020
#
# Deep Learning with Python
# Chapter 3: Getting started with neural networks
# 3.6-classifying-newswires.ipynb
# The Reuters dataset
################################################ 

# In this section, we will build a network to classify Reuters newswires into 46 different mutually-exclusive topics. 
# Since we have many classes, this problem is an instance of "multi-class classification", 
# and since each data point should be classified into only one category, the problem is more specifically an instance of "single-label, multi-class classification". 
# If each data point could have belonged to multiple categories (in our case, topics) then we would be facing a "multi-label, multi-class classification" problem.

# Like IMDB and MNIST, the Reuters dataset comes packaged as part of Keras. Let's take a look right away:

import keras
# from tensorflow import keras    # TensorFlow 2.0   
keras.__version__

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# We have 8,982 training examples and 2,246 test examples:
print("len(train_data)",len(train_data))
print("len(test_data)",len(test_data))
print("train_data[10]= ",train_data[10])

# Here's how you can decode it back to words:
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print("decoded_newswire= ",decoded_newswire)

# The label associated with an example is an integer between 0 and 45: a topic index.
print("train_labels[10]= ",train_labels[10])


## Preparing the data
# We can vectorize the data as the follows:
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# To vectorize the labels, there are two possibilities: we could just cast the label list as an integer tensor, or we could use a "one-hot" encoding. 
# One-hot encoding is a widely used format for categorical data, also called "categorical encoding". 
# For a more detailed explanation of one-hot encoding, you can refer to Chapter 6, Section 1. 
# In our case, one-hot encoding of our labels consists in embedding each label as an all-zero vector with a 1 in the place of the label index, e.g.:
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)

# Note that there is a built-in way to do this in Keras, which you have already seen in action in our MNIST example:
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


## Building our network
# This topic classification problem looks very similar to our previous movie review classification problem: in both cases, we are trying to classify short snippets of text. 
# There is however a new constraint here: the number of output classes has gone from 2 to 46, i.e. the dimensionality of the output space is much larger.

# In a stack of Dense layers like what we were using, each layer can only access information present in the output of the previous layer. 
# If one layer drops some information relevant to the classification problem, this information can never be recovered by later layers: each layer can potentially become an "information bottleneck". 
# In our previous example, we were using 16-dimensional intermediate layers, but a 16-dimensional space may be too limited to learn to separate 46 different classes: 
# such small layers may act as information bottlenecks, permanently dropping relevant information.

#For this reason we will use larger layers. Let's go with 64 units:
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# There are two other things you should note about this architecture:
# We are ending the network with a Dense layer of size 46. This means that for each input sample, our network will output a 46-dimensional vector. 
# Each entry in this vector (each dimension) will encode a different output class.
# The last layer uses a softmax activation. 
# It means that the network will output a probability distribution over the 46 different output classes, 
# i.e. for every input sample, the network will produce a 46-dimensional output vector where output[i] is the probability that the sample belongs to class i. 
# The 46 scores will sum to 1.

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# The best loss function to use in this case is categorical_crossentropy. It measures the distance between two probability distributions: 
# in our case, between the probability distribution output by our network, and the true distribution of the labels. 
# By minimizing the distance between these two distributions, we train our network to output something as close as possible to the true labels.     


## Validating our approach
# Let's set apart 1,000 samples in our training data to use as a validation set:
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Now let's train our network for 20 epochs:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Note that the call to model.fit() returns a History object. This object has a member history, 
# which is a dictionary containing data about everything that happened during training. Let's take a look at it:
history_dict = history.history
print("history_dict.keys()",history_dict.keys())
# Output: history_dict.keys() dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# Let's display its loss and accuracy curves:
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()   # clear figure
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# It seems that the network starts overfitting after 8 epochs. 
# Let's train a new network from scratch for 8 epochs, then let's evaluate it on the test set:
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print("8 epochs/64-dimensional results: ",results)

# Our approach reaches an accuracy of ~78%. 
# With a balanced binary classification problem, the accuracy reached by a purely random classifier would be 50%, 
# but in our case it is closer to 19%, so our results seem pretty good, at least when compared to a random baseline:
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)


## Generating predictions on new data
# We can verify that the predict method of our model instance returns a probability distribution over all 46 topics. 
# Let's generate topic predictions for all of the test data:

predictions = model.predict(x_test)

# Each entry in predictions is a vector of length 46:
print("predictions[0].shape: ",predictions[0].shape)

# The coefficients in this vector sum to 1:
print("np.sum(predictions[0]): ",np.sum(predictions[0]))

# The largest entry is the predicted class, i.e. the class with the highest probability:
print("np.argmax(predictions[0]): ",np.argmax(predictions[0]))


## A different way to handle the labels and the loss
# We mentioned earlier that another way to encode the labels would be to cast them as an integer tensor, like such:
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# The only thing it would change is the choice of the loss function. 
# Our previous loss, categorical_crossentropy, expects the labels to follow a categorical encoding. 
# With integer labels, we should use sparse_categorical_crossentropy:
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
# This new loss function is still mathematically the same as categorical_crossentropy; it just has a different interface.


## On the importance of having sufficiently large intermediate layers
# We mentioned earlier that since our final outputs were 46-dimensional, we should avoid intermediate layers with much less than 46 hidden units. 
# Now let's try to see what happens when we introduce an information bottleneck by having intermediate layers significantly less than 46-dimensional, e.g. 4-dimensional.
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)          
print("20 epochs/4-dimensional results: ",results)

# Our network now seems to peak at ~71% test accuracy, a 8% absolute drop. 
# This drop is mostly due to the fact that we are now trying to compress a lot of information
#  (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. 
# The network is able to cram most of the necessary information into these 8-dimensional representations, but not all of it.


## Further experiments
# 1) Try using larger or smaller layers: 32 units, 128 units...
# 2) We were using two hidden layers. Now try to use a single hidden layer, or three hidden layers.


## Take away:
# 1) If you are trying to classify data points between N classes, your network should end with a Dense layer of size N.
# 2) In a single-label, multi-class classification problem, your network should end with a softmax activation, 
# so that it will output a probability distribution over the N output classes.
# 3) Categorical crossentropy is almost always the loss function you should use for such problems. 
# It minimizes the distance between the probability distributions output by the network, and the true distribution of the targets.
# 4) There are two ways to handle labels in multi-class classification: 
# ** Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using categorical_crossentropy as your loss function. 
# ** Encoding the labels as integers and using the sparse_categorical_crossentropy loss function.
# 5) If you need to classify data into a large number of categories, 
# then you should avoid creating information bottlenecks in your network by having intermediate layers that are too small.















