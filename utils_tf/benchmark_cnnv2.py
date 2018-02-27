"""Builds a convolutional neural network with a flexible number of
convolutional and pooling layers and runs a benchmark by training it on
a synthetic dataset
"""

import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import timeline
from utils_tf.utils_cnn import cnn_multidevicev3, average_gradients
from utils_tf.utils_data import build_dataset

def benchmark_cnn(
        num_layers,
        num_features,
        kernel_size,
        pooling_size,
        fully_connected_size,
        lr_initial,
        lr_decay,
        precision,
        num_trainimg,
        num_testimg,
        imgsize,
        numsteps,
        batchsize,
        data_in_mem,
        logstep,
        trackingstep,
        num_gpu,
        devlist,
        data_file,
        train_dir):

    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    numdev = len(devlist)

    datatype = "float%d" %precision


    if data_file=='':
        data_in_mem = True
        gen_data = True
        num_channels = 1
        num_classes = 2

    else:
        gen_data = False
        imgsize = 32
        num_channels = 3
        num_classes = 10


    # Generate the dataset and build the queue
    with tf.device('/cpu:0'):
        if data_in_mem:
            if gen_data==True:
                trainimg, trainlabel, testimg, testlabel = build_dataset.generate_data(
                        num_trainimg,
                        num_testimg,
                        imgsize,
                        datatype)
            elif gen_data==False:
                trainimg, trainlabel = build_dataset.load_full_dataset(
                        data_file,
                        imgsize,
                        imgsize)

            # Generate tf.data dataset
            train_data = tf.data.Dataset.from_tensor_slices((trainimg, trainlabel))
            # Repeat data indefinetely
            train_data = train_data.repeat()
            # Shuffle data
            train_data = train_data.shuffle(5*batchsize)
            # Prepare batches
            train_batch = train_data.batch(batchsize)
            # Create an iterator
            iterator = train_batch.make_one_shot_iterator()
        else:
            filenames = tf.placeholder(tf.string, shape=[None])
            iterator = build_dataset.get_iterator(
                    filenames,
                    batchsize,
                    imgsize,
                    imgsize,
                    num_channels,
                    numdev)


    # Set learning rate and build optimizer
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0),
        trainable=False)

    lr = numdev*tf.train.exponential_decay(
            lr_initial,
            global_step,
            5000,
            lr_decay,
            staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)


    # Build the model
    tower_gradients = []

    for dev_ind in range(numdev):
        dev = devlist[dev_ind]
        print("device %s" % dev)
        with tf.device(devlist[dev_ind]):
            with tf.name_scope('tower_%d' % (dev_ind)) as scope:
                images,labels = iterator.get_next()
                loss, logits = cnn_multidevicev3.build_model(
                        images,
                        labels,
                        num_layers,
                        num_features,
                        kernel_size,
                        pooling_size,
                        fully_connected_size,
                        num_channels,
                        num_classes,
                        datatype)
                tf.get_variable_scope().reuse_variables()

                gradient = optimizer.compute_gradients(loss)
                tower_gradients.append(gradient)
    with tf.device('/cpu:0'):
        mean_gradient = average_gradients.average_gradients(tower_gradients)
        train_op = optimizer.apply_gradients(mean_gradient, global_step=global_step)
        loss_summary = tf.summary.scalar("training_loss", loss)


    acc = np.empty([numsteps,1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not data_in_mem:
            sess.run(it.initializer, feed_dict={filenames: data_file})
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        writer = tf.summary.FileWriter(train_dir, sess.graph, flush_secs=600)
        t_train = time.time()
        t_step = time.time()
        for i in range(numsteps):
            _, loss_summ = sess.run(
                    [train_op, loss_summary],
                    options=options,
                    run_metadata=run_metadata)

            if logstep > 0:
                if i%logstep==0:
                    writer.add_summary(loss_summ, i)
                if i>0 and i%(trackingstep)==0:
                    t = time.time()
                    print("%.2f sec, step %d, %.2f images/sec" %(
                            time.time()-t_train,
                            i,
                            trackingstep*batchsize*numdev/(t-t_step)))
                    t_step = t
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('%s/timeline_step_%d.json' % (train_dir,i), 'w') as f:
                        f.write(chrome_trace)

        timeUsed_train = time.time()-t_train

        t_infer = time.time()
        loss_validation = sess.run(loss)
        timeUsed_infer = time.time() - t_infer
        print("After %d steps: loss = %.2f" %(numsteps, loss_validation))

    return timeUsed_train, timeUsed_infer
