"""Function to build a synthetic dataset of images"""

import numpy as np
import pickle
import os

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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def import_cifar(data_dir):
    trainimg = np.empty([50000,3072])
    lbl = np.empty(50000)
    trainlabel = np.zeros([50000,10])
    for i in range(1,6):
        filename = os.path.join(data_dir, 'data_batch_%d' % i)
        d = unpickle(filename)
        trainimg[(i-1)*10000:i*10000,:] = d[b'data']
        lbl[(i-1)*10000:i*10000] = d[b'labels']
    trainimg = trainimg.reshape([50000,3,32,32])
    trainimg = np.transpose(trainimg, (0, 2, 3, 1))
    trainlabel[np.arange(50000), np.int8(lbl)] = 1

    testlabel = np.zeros([10000,10])
    filename = os.path.join(data_dir, 'test_batch')
    d = unpickle(filename)
    testimg = d[b'data']
    lbl = d[b'labels']
    testimg = testimg.reshape([10000,3,32,32])
    testimg = np.transpose(testimg, (0, 2, 3, 1))
    testlabel[np.arange(10000), np.int8(lbl)] = 1

    return trainimg, trainlabel, testimg, testlabel
