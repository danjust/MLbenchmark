"""Benchmark matrix multiplication"""

import tensorflow as tf
import time


def benchmark_matmul(n,iterations,devlist,datatype):
    devlist = devlist.split(',')
    datatype = eval('tf.%s' %(datatype))
    for dev in devlist:
        with tf.device(dev):
            matA = tf.Variable(tf.ones([n,n],dtype=datatype))
            matB = tf.Variable(tf.ones([n,n],dtype=datatype))
            prod = tf.matmul(matA,matB)

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)
                    )
            )
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Warm-up run
    sess.run(prod.op)

    # Benchmark run
    t = time.time()
    for _ in range(iterations):
        sess.run(prod.op)
    return (time.time()-t)/iterations
