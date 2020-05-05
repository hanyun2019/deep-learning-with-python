################################################### 
# Haowen modified on May 2, 2020
#
# Deep Learning with Python
# Chapter 7: Advanced deep-learning best practices
# 
# Multi-output models
# 
################################################### 

from keras import layers
from keras import Input 
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

age_targets = {24, 56, 34, 87, 22, 43, 65, 23, 54, 26}
income_targets = {30, 300, 120, 500, 50, 150, 60, 80, 320, 80}
gender_targets = {1,0,1,0,1,0,1,0,1,0}

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop', 
              loss={'age': 'mse', 
                    'income': 'categorical_crossentropy', 
                    'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25,
                            'income': 1.,
                            'gender': 10.})

model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets},
        epochs=10, batch_size=64)
