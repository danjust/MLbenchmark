"""Benchmark matrix multiplication"""

import tensorflow as tf
import numpy as np
import time


def benchmark_matmul(n,iterations,logFLOPs,num_gpu,devlist,precision):
    # generate list of devices if devlist is empty
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    ops = n**3 + (n-1)*n**2
    if FLOPs>0:
        iterations = int(np.ceil(10^targetFLOPs/ops))
        print("Running %d iterations" %iterations)

    datatype = eval('tf.float%d' %(precision))

    for dev in devlist:
        with tf.device(dev):
            matA = tf.Variable(tf.ones([n,n],dtype=datatype))
            matB = tf.Variable(tf.ones([n,n],dtype=datatype))
            prod = tf.matmul(matA,matB)

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)),
            log_device_placement=False)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Warm-up run
        sess.run(prod.op)

        # Benchmark run
        t = time.time()
        for _ in range(iterations):
            sess.run(prod.op)
    return (time.time()-t)/iterations
