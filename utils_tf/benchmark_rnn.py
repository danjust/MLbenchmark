"""Benchmark recurrent neural networks
choose 'rnn' or 'lstm' for basic recurrent neural network cell
or basic long short-term memory cell"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def benchmark_rnn(
        rnn_type,
        seq_length,
        batch_size,
        num_samples,
        num_units,
        num_classes,
        learning_rate,
        iterations,
        num_gpu,
        devlist,
        precision):

    # generate list of devices if devlist is empty
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    datatype = eval('np.float%d' %(precision))

    # Generate synthetic data
    data = np.random.rand(num_samples,seq_length).astype(datatype)
    target = np.zeros([num_samples,num_classes])
    target[[ix for ix in range(num_samples)], np.random.randint(num_classes)] = 1

    # Initialize graph
    for dev in devlist:
        with tf.device(dev):
            tf.reset_default_graph()
            X = tf.placeholder("float", [None, seq_length])
            Y = tf.placeholder("float", [None, num_classes])
            weights = tf.Variable(tf.random_normal([num_units, num_classes]))
            biases = tf.Variable(tf.random_normal([num_classes]))

            if rnn_type=='rnn' or rnn_type=='RNN':
                cell = rnn.BasicRNNCell(num_units)
            elif rnn_type=='lstm' or rnn_type=='LSTM':
                cell = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
            else:
                raise Exception('Unknown rnn type %s, must be rnn or lstm '%(network_type))

            outputs, states = rnn.static_rnn(cell, [X], dtype=tf.float32)

            logits = tf.matmul(outputs[-1], weights) + biases
            prediction = tf.nn.softmax(logits)

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits,
                            labels=Y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op)

    # Creates the session
    config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)
                    )
            )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Warm-up run
        batchuse = np.random.randint(0,num_samples,batch_size)
        batch_x = data[batchuse,:]
        batch_y = target[batchuse,:]
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Benchmark run
        t_start = time.time()
        for step in range(iterations):
            batchuse = np.random.randint(0,num_samples,batch_size)
            batch_x = data[batchuse,:]
            batch_y = target[batchuse,:]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

    return((time.time()-t_start)/iterations)
