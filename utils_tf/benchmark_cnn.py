"""Builds a convolutional neural network with a flexible number of
convolutional and pooling layers and runs a benchmark by training it on
a synthetic dataset
"""

import tensorflow as tf
import numpy as np
import time
from utils_tf.utils_cnn import cnn_multidevice, build_dataset

def benchmark_cnn(
        num_layers,
        num_features,
        kernelsize,
        poolingsize,
        fully_connected_size,
        lr_initial,
        lr_decay,
        num_trainimg,
        num_testimg,
        imgsize,
        numsteps,
        batchsize,
        logstep,
        num_gpu,
        devlist,
        data_dir,
        train_dir):

    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    if data_dir=='':
        gen_data=True
        num_channels=1
        num_classes=2
    else:
        gen_data=False
        imgsize=32
        num_channels=3
        num_classes=10

    # batchsize is per GPU
    total_batchsize = int(batchsize*len(devlist))

    # Generate the Graph
    g, x, y_, train_op, loss_summary, accuracy_summary, lr_summary, accuracy = cnn_multidevice.build_graph(
            num_layers,
            num_features,
            kernelsize,
            poolingsize,
            lr_initial,
            lr_decay,
            fully_connected_size,
            imgsize,
            num_channels,
            num_classes,
            devlist)


    # Generate the dataset
    if gen_data==True:
        trainimg, trainlabel, testimg, testlabel = build_dataset.build_dataset(
                num_trainimg,
                num_testimg,
                imgsize)
    elif gen_data==False:
        trainimg, trainlabel, testimg, testlabel = build_dataset.import_cifar(data_dir)

    acc = np.empty([numsteps,1])

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(train_dir, sess.graph, flush_secs=60, max_queue=250)
        t_train = time.time()
        for i in range(numsteps):
            batch = np.random.randint(
                    0,
                    np.size(trainlabel,0),
                    total_batchsize)
            img_batch = trainimg[batch,:,:]
            label_batch = trainlabel[batch,:]
            _, loss_summ, acc_summ, lr_summ, acc[i] = sess.run([train_op, loss_summary, accuracy_summary, lr_summary,accuracy], feed_dict={x: img_batch, y_: label_batch})
            writer.add_summary(loss_summ, i)
            writer.add_summary(acc_summ, i)
            writer.add_summary(lr_summ, i)
            if logstep > 0:
                if i%logstep==0:
                    print("%.2f sec, step %d: accuracy = %.2f" %(time.time()-t_train, i, acc[i]))

        timeUsed_train = time.time()-t_train

        t_infer = time.time()
        acc_validation = sess.run(accuracy,feed_dict={x: testimg, y_: testlabel})
        timeUsed_infer = time.time() - t_infer
        print("After %d steps: accuracy = %.2f" %(numsteps, acc_validation))

    return timeUsed_train, timeUsed_infer
