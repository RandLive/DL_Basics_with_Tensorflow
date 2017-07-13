# -*- coding: utf-8 -*-

import tensorflow as tf
# Remove the previous weights and bias
# 移除之前的权重和偏置项
tf.reset_default_graph()

save_file = './model.ckpt'

# Two Variables: weights and bias
# 两个变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
# 用来存取 Tensor 变量的类
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    # 加载权重和偏置项
    
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    # 显示权重和偏置项
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))