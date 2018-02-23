"""Benchmark convolution"""

import tensorflow as tf
import time


def benchmark_conv(n,kernelsize,iterations,num_gpu,devlist,precision):
    # generate list of devices if devlist is empty
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    datatype = eval('tf.float%d' %(precision))

    for dev in devlist:
        with tf.device(dev):
            matA = tf.Variable(tf.ones([1,n,n,1],dtype=datatype))
            kernel = tf.Variable(
                    tf.ones([kernelsize,kernelsize,1,1],
                    dtype=datatype))
            conv = tf.nn.conv2d(
                    input=matA,filter=kernel,
                    strides=[1,1,1,1],
                    padding="VALID")

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)),
            log_device_placement=False)


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Warm-up run
        sess.run(conv.op)

        # Benchmark run
        t = time.time()
        for _ in range(iterations):
            sess.run(conv.op)
    return (time.time()-t)/iterations
