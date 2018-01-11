"""Builds a convolutional neural network with a flexible number of
convolutional and pooling layers and runs a benchmark by training it on
a synthetic dataset
"""

import tensorflow as tf
import numpy as np
import time
from utils_tf.utils_cnn import build_cnn, build_dataset

def benchmark_cnn(
        num_layers,
        num_features,
        kernelsize,
        poolingsize,
        num_trainimg,
        num_testimg,
        imgsize,
        numsteps,
        batchsize,
        logstep):

    # Generate the Graph
    g, x, y_ , train_op, accuracy = build_cnn.build_graph(
            num_layers,
            num_features,
            kernelsize,
            poolingsize,
            imgsize)

    # Generate the dataset
    trainimg, trainlabel, testimg, testlabel = build_dataset.build_dataset(
            num_trainimg,
            num_testimg,
            imgsize)

    acc = np.empty([numsteps,1])

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        t_train = time.time()
        for i in range(numsteps):
            batch = np.random.randint(0,num_trainimg,batchsize)
            acc[i] = accuracy.eval(feed_dict={x: trainimg[batch,:,:], y_: trainlabel[batch,:]})
            if logstep > 0:
                if i%logstep==0:
                    print("%.2f sec, step %d, accuracy = %.2f" %(time.time()-t_train, i, acc[i]))
            sess.run(train_op,feed_dict={x: trainimg[batch,:,:], y_: trainlabel[batch,:]})
        timeUsed_train = time.time()-t_train

        t_infer = time.time()
        acc_validation = accuracy.eval(feed_dict={x: testimg, y_: testlabel})
        timeUsed_infer = time.time() - t_infer
        print("After %d steps: accuracy = %.2f" %(numsteps, acc_validation))

    return timeUsed_train, timeUsed_infer, acc
