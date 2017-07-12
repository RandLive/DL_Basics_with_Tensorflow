# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:14:34 2017

@author: dream_rab04is

In this example, the memory size can be calculated. For large
dataset, GPUs are needed, otherwise, mini-batching needs to be
performed.

"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)



with tf.device('/gpu:1'):

    # Import MNIST data
    mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

    # The features are already scaled and the data is shuffled
    train_features = mnist.train.images
    test_features = mnist.test.images

    train_labels = mnist.train.labels.astype(np.float32)
    test_labels = mnist.test.labels.astype(np.float32)

    # Weights & bias
    weights = tf.Variable(tf.random_normal([n_input, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))