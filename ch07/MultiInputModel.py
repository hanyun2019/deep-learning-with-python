################################################### 
# Haowen modified on May 2, 2020
#
# Deep Learning with Python
# Chapter 7: Advanced deep-learning best practices
# 
# Multi-input models
# 
################################################### 

# A typical question-answering model has two inputs: a natural-language question and a text snippet (such as a news article) providing information to be used for answering the question. 
# The model must then produce an answer: in the simplest pos- sible setup, this is a one-word answer obtained via a softmax over some predefined vocabulary.

from keras.models import Model
from keras import layers
from keras import Input
from keras.utils import to_categorical

import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
emcoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
emcoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([emcoded_text, emcoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length)) 

answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = to_categorical(answers, answer_vocabulary_size)

model.fit([text, question], answers, epochs=10, batch_size=128)
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)



