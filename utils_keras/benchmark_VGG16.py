import keras
import numpy as np
import time

def benchmark_VGG16(
        imgwidth,
        imghight,
        numclasses,
        optimizer,
        iterations,
        batchsize,
        precision,
        logfile):

    # Generate synthetic data
    datatype = eval('np.float%d' %(precision))
    batch_data = np.zeros(
            [batchsize,imgwidth,imghight,3],
            dtype=datatype)
    batch_label = np.zeros(
            [batchsize,numclasses],
            dtype=np.int16)

    # Define model
    model = keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=[imgwidth,imghight,3],
            pooling=None,
            classes=numclasses)

    # Define optimizer
    if optimizer=='sgd':
        opt = keras.optimizers.SGD(lr=0.01)
    elif optimizer=='rmsprop':
        opt = keras.optimizers.rmsprop(lr=0.0001)

    # Compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # Run warm-up
    model.train_on_batch(batch_data,batch_label)

    # Run benchmark
    t_start = time.time()
    for i in range(iterations):
        model.train_on_batch(batch_data,batch_label)
    dur = time.time()-t_start

    img_per_sec = (iterations*batchsize)/dur

    logtext = ('VGG-16, %d, %d, %d, %.3f\n'
    %(imgwidth,precision,batchsize,img_per_sec))
    f = open(logfile,'a+')
    f.write(logtext)
    f.close()

    return img_per_sec
