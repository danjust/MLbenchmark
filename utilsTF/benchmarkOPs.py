import tensorflow as tf
import time


def Benchmark_MatMul(n,iterations,dev):
    with tf.device(dev):
        matA = tf.Variable(tf.ones([n,n],dtype=tf.float32))
        matB = tf.Variable(tf.ones([n,n],dtype=tf.float32))
        prod = tf.matmul(matA,matB)

    # Creates the session
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Warm-up run
    sess.run(prod.op)

    # Benchmark run
    t = time.time()
    for _ in range(iterations):
        sess.run(prod.op)
    return (time.time()-t)/iterations


def Benchmark_Conv(n,kernelsize,iterations,dev):
    with tf.device(dev):
        matA = tf.Variable(tf.ones([1,n,n,1],dtype=tf.float32))
        kernel = tf.Variable(tf.ones([kernelsize,kernelsize,1,1],dtype=tf.float32))
        conv = tf.nn.conv2d(input=matA,filter=kernel,strides=[1,1,1,1],padding="SAME")

    # Creates the session
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Warm-up run
    sess.run(conv.op)

    # Benchmark run
    t = time.time()
    for _ in range(iterations):
        sess.run(conv.op)
    return (time.time()-t)/iterations
