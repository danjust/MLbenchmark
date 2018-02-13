"""Function to build a synthetic dataset of images"""

import numpy as np
import tensorflow as tf
import os
import sys

def build_dataset(num_trainimg,num_testimg,imgsize):
    trainimg = np.zeros([num_trainimg,imgsize,imgsize,1]).astype(np.float32)
    trainlabel = np.zeros([num_trainimg,2]).astype(np.int16)
    for i in range(num_trainimg):
        isvert = np.random.randint(2)
        line = np.random.randint(imgsize)
        if isvert==1:
            trainimg[i,:,line,0] = 1
            trainlabel[i,0] = 1
        else:
            trainimg[i,line,:,0] = 1
            trainlabel[i,1] = 1

    testimg = np.zeros([num_testimg,imgsize,imgsize,1]).astype(np.float32)
    testlabel = np.zeros([num_testimg,2]).astype(np.int16)
    for i in range(num_testimg):
        isvert = np.random.randint(2)
        line = np.random.randint(imgsize)
        if isvert==1:
            testimg[i,:,line,0] = 1
            testlabel[i,0] = 1
        else:
            testimg[i,line,:,0] = 1
            testlabel[i,1] = 1

    return trainimg, trainlabel, testimg, testlabel


def unpickle3(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle2(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def unpickle(file):
    if sys.version_info.major==3:
        dict = unpickle3(file)
    elif sys.version_info.major==2:
        dict = unpickle2(file)
    return dict

def import_cifar(data_dir):
    img = np.empty([50000,3072])
    lbl = np.empty(50000)
    trainlabel = np.zeros([50000,10])

    # Build queue
    queue = tf.RandomShuffleQueue(capacity=50000,min_after_dequeue=10000, dtypes=[tf.float32,tf.int16],shapes=[[50000,32,32,3],[50000,10]])
    for i in range(1,6):
        filename = os.path.join(data_dir, 'data_batch_%d' % i)
        d = unpickle(filename)
        img[(i-1)*10000:i*10000,:] = d[b'data']
        lbl[(i-1)*10000:i*10000] = d[b'labels']

    trainimg = img.reshape([50000,3,32,32])
    trainimg = np.transpose(trainimg, (0, 2, 3, 1))
    trainlabel[np.arange(50000), np.int8(lbl)] = 1

    # enqueue dataset
    enqueue_op = queue.enqueue([trainimg,trainlabel])




    testlabel = np.zeros([10000,10])
    filename = os.path.join(data_dir, 'test_batch')
    d = unpickle(filename)
    testimg = d[b'data']
    lbl = d[b'labels']
    testimg = testimg.reshape([10000,3,32,32])
    testimg = np.transpose(testimg, (0, 2, 3, 1))
    testlabel[np.arange(10000), np.int8(lbl)] = 1

    return queue, enqueue_op, testimg, testlabel
