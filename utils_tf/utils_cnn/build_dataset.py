"""Function to build a synthetic dataset of images"""

import numpy as np

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
