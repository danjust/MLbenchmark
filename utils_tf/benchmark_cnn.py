"""Builds a convolutional neural network with a flexible number of
convolutional and pooling layers and runs a benchmark by training it on
a synthetic dataset
"""

import tensorflow as tf
import numpy as np
import time
from utils_tf.utils_cnn import build_cnn_multdevice, build_dataset

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
        logstep,
        num_gpu,
        devlist):

    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    # Generate the Graph
    g, x, y_ , train_op, loss, accuracy, prediction = build_cnn_multdevice.build_graph(
            num_layers,
            num_features,
            kernelsize,
            poolingsize,
            imgsize,
            devlist)

    # Generate the dataset
    trainimg, trainlabel, testimg, testlabel = build_dataset.build_dataset(
            num_trainimg,
            num_testimg,
            imgsize)

    loss_step = np.empty([numsteps,1])
    acc = np.empty([numsteps,1])

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        t_train = time.time()
        for i in range(numsteps):
            batch = np.random.randint(0,num_trainimg,batchsize*len(devlist))
            _, loss_step[i], acc[i] = sess.run([train_op,loss,accuracy],feed_dict={x: trainimg[batch,:,:], y_: trainlabel[batch,:]})
            if logstep > 0:
                if i%logstep==0:
                    print("%.2f sec, step %d, loss = %.2f, accuracy = %.2f" %(time.time()-t_train, i, loss_step[i],acc[i]))

        timeUsed_train = time.time()-t_train

        t_infer = time.time()
        acc_validation,prediction_validation = sess.run([accuracy,prediction],feed_dict={x: testimg, y_: testlabel})
        timeUsed_infer = time.time() - t_infer
        print("After %d steps: accuracy = %.2f" %(numsteps, acc_validation))

    return timeUsed_train, timeUsed_infer, loss_step
