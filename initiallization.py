import tensorflow as tf

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
