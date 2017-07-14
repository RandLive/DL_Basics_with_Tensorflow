# -*- coding: utf-8 -*-
"""
TensorFlow 提供了 tf.nn.conv2d() 和 tf.nn.bias_add() 函数来创建你自己的卷积层。
上述代码用了 tf.nn.conv2d() 函数来计算卷积，weights 作为滤波器，[1, 2, 2, 1] 作为 strides。
TensorFlow 对每一个 input 维度使用一个单独的 stride 参数，[batch, input_height, input_width, input_channels]。
我们通常把 batch 和 input_channels （strides 序列中的第一个第四个）的 stride 设为 1。
你可以专注于修改 input_height 和 input_width， batch 和 input_channels 都设置成 1。
input_height 和 input_width strides 表示滤波器在input 上移动的步长。上述例子中，在 input 之后，设置了一个 5x5 ，stride 为 2 的滤波器。
tf.nn.bias_add() 函数对矩阵的最后一维加了偏置项。
"""

import tensorflow as tf

# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)