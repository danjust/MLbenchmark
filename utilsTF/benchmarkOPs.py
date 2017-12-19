import tensorflow as tf
import time


def Benchmark_MatMul(n,iterations,dev):

    with tf.device(dev):
        matA = tf.Variable(tf.ones([n,n],dtype=tf.float32))
        matB = tf.Variable(tf.ones([n,n],dtype=tf.float32))
        prod = tf.matmul(matA,matB)

    # Creates a session with log_device_placement set to True.
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Runs the op.
    sess.run(prod.op)
    t = time.time()
    for _ in range(iterations):
        sess.run(prod.op)
    return (time.time()-t)/iterations


def Benchmark_Conv(n,kernelsize,iterations,dev):

    with tf.device(dev):
        matA = tf.Variable(tf.ones([1,n,n,1],dtype=tf.float32))
        kernel = tf.Variable(tf.ones([kernelsize,kernelsize,1,1],dtype=tf.float32))
        conv = tf.nn.conv2d(input=matA,filter=kernel,strides=[1,1,1,1],padding="SAME")

    # Creates a session with log_device_placement set to True.
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Runs the op.
    sess.run(conv.op)
    t = time.time()
    for _ in range(iterations):
        sess.run(conv.op)
    return (time.time()-t)/iterations
