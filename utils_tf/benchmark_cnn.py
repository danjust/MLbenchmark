"""A benchmark of ML hardware by training a convolutional neural network using
the CIFAR-10 dataset. This script is based on
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10.
"""

import time
import os

import tensorflow as tf

from utils_tf.utils_cifar10 import cifar10, cifar10_input


def train(data_dir,batch_size,max_steps,num_gpu,devlist):
    # generate list of devices if devlist is empty
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes
        # ending up on GPU and resulting in a slow down.
        for dev in devlist:
            with tf.device(dev):
                binary_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
                images, labels = cifar10_input.inputs(eval_data=False,
                                                    data_dir=binary_dir,
                                                    batch_size=batch_size)


                # Build a Graph that computes the logits predictions from the
                # inference model.
                logits = cifar10.inference(images,batch_size)

                # Calculate loss.
                loss = cifar10.loss(logits, labels)

                # Build a Graph that trains the model with one batch of examples and
                # updates the model parameters.
                train_op = cifar10.train(loss, global_step,batch_size)

        with tf.train.MonitoredTrainingSession() as mon_sess:
            print("========================================\n")
            print("start training cnn")
            tstart = time.time()
            for steps in range(max_steps):
                mon_sess.run(train_op)
            tstop = time.time()
            dur = tstop-tstart
            print("========================================\n")
    return dur
