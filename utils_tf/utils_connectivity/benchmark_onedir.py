import tensorflow as tf
import time

def benchmark_onedir(
        host_dev,
        remote_dev,
        precision,
        size_x,
        size_y,
        iterations):

    dtype = eval('tf.float%d' %(precision))

    with tf.device(host_dev):
        var_host = tf.Variable(tf.ones([size_x,size_y],dtype=dtype))

    with tf.device(remote_dev):
        var_remote = tf.identity(var_host)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(var_remote)

        t = time.time()
        for _ in range(iterations):
            sess.run(var_remote)
        timeUsed = (time.time()-t)/iterations

    return timeUsed
