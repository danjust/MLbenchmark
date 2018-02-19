import tensorflow as tf
import numpy as np
import time


def benchmark_latency(iterations,device1,device2):
    with tf.device(device1):
        data = tf.data.Dataset.range(1)
        data = data.repeat()
        iterator = data.make_one_shot_iterator()

    with tf.device(device2):
        new_val = iterator.get_next()

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)),
            log_device_placement=False)
    with tf.Session(config=config) as sess:

        # Warm-up run
        sess.run(mat_device.op)

        # Benchmark run
        t = time.time()
        for _ in range(iterations):
            sess.run(new_val.op)
    return (time.time()-t)/iterations
