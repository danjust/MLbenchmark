"""Function to build a synthetic dataset of images"""

import numpy as np
import tensorflow as tf
import os
import sys

def build_dataset(num_trainimg,num_testimg,imgsize,dtype):
    trainimg = np.zeros([num_trainimg,imgsize,imgsize,1]).astype(eval('np.%s' %(dtype)))
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

    train_data=tf.data.Dataset.from_tensor_slices((trainimg, trainlabel))

    testimg = np.zeros([num_testimg,imgsize,imgsize,1]).astype(eval('np.%s' %(dtype)))
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

    test_data=tf.data.Dataset.from_tensor_slices((testimg, testlabel))

    return train_data, test_data


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
    img = np.empty([50000,3072],dtype=astype(eval('np.%s' %(dtype))))
    lbl = np.empty(50000,dtype=np.int16)
    trainlabel = np.zeros([50000,10],dtype=np.int16)

    for i in range(1,6):
        filename = os.path.join(data_dir, 'data_batch_%d' % i)
        d = unpickle(filename)
        img[(i-1)*10000:i*10000,:] = d[b'data']
        lbl[(i-1)*10000:i*10000] = d[b'labels']

    trainimg = img.reshape([50000,3,32,32])
    trainimg = np.transpose(trainimg, (0, 2, 3, 1))
    trainlabel[np.arange(50000), np.int8(lbl)] = 1

    train_data=tf.data.Dataset.from_tensor_slices((trainimg, trainlabel))


    testlabel = np.zeros([10000,10])
    filename = os.path.join(data_dir, 'test_batch')
    d = unpickle(filename)
    testimg = d[b'data']
    lbl = d[b'labels']
    testimg = testimg.reshape([10000,3,32,32])
    testimg = np.transpose(testimg, (0, 2, 3, 1))
    testlabel[np.arange(10000), np.int8(lbl)] = 1

    test_data=tf.data.Dataset.from_tensor_slices((testimg, testlabel))

    return train_data, test_data
