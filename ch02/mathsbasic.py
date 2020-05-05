# Haowen modified on Jan 1, 2020
#
# Deep Learning with Python
# Chapter 2: Before we begin: the mathematical building blocks of neural networks 
#

# My MacBook environment:
# $ pwd
# /Users/ML/deeplearning/dl-with-python/deep-learning-with-python-notebooks
# $ jupyter notebook

# $ python
# Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)] on darwin
# >>> import tensorflow
# >>> tensorflow.__version__
# '2.0.0'
# >>> import keras
# Using TensorFlow backend.
# >>> keras.__version__
# '2.3.1'

# TF2.0 is based on Keras. If you choose this book to study TF2.0, remember to replace:
# import keras -> from tensorflow import keras

# Chapter 2: Before we begin: the mathematical building blocks of neural networks 
# At its core, a tensor is a container for data—almost always numerical data. So, it’s a container for numbers. 
# You may be already familiar with matrices, which are 2D tensors: 
# tensors are a generalization of matrices to an arbitrary number of dimensions 
# (note that in the context of tensors, a dimension is often called an axis).
#

# Scalars (0D tensors)
import numpy as np
x = np.array(12)
print("Scalars(0D tensors): x=",x)

# Vectors (1D tensors)
x = np.array([12, 3, 6, 14])
print("Vectors(1D tensors): x=",x)
print("Vectors(1D tensors): x.ndim=",x.ndim)

# Matrices (2D tensors)
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print("Matrices (2D tensors): x=",x)
print("Matrices (2D tensors): x.ndim=",x.ndim)

# 3D tensors and higher-dimensional tensors
x = np.array([[[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]]])
print("Matrices (3D tensors): x=\n",x)
print("Matrices (3D tensors): x.ndim=",x.ndim)





