import tensorflow as tf
import numpy as np
import time


def benchmark_latency(iterations,device1,device2):
    with tf.device(device1):
        data = 1

    with tf.device(device2):
        new_val = tf.identity(data)

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)),
            log_device_placement=False)
    with tf.Session(config=config) as sess:

        # Warm-up run
        sess.run(new_val.op)

        # Benchmark run
        t = time.time()
        for _ in range(iterations):
            sess.run(new_val.op)
    return (time.time()-t)/iterations
