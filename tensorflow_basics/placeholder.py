# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 09:19:15 2017

@author: ML
"""

# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x, feed_dict={x:123})

    return output

print(run())