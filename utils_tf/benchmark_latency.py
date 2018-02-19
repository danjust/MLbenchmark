import tensorflow as tf
import numpy as np
import time


def benchmark_latency(matsize,iterations,device):
    with tf.device('/cpu:0'):
        mat = tf.Variable(tf.ones([matsize,matsize],dtype=tf.float32))
    with tf.device(device):
        mat_device = tf.identity(mat)

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
            sess.run(mat_device.op)
    return (time.time()-t)/iterations
