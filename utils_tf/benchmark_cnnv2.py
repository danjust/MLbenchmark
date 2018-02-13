"""Builds a convolutional neural network with a flexible number of
convolutional and pooling layers and runs a benchmark by training it on
a synthetic dataset
"""

import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import timeline
from utils_tf.utils_cnn import cnn_multidevicev2, build_datasetv2, average_gradients

def benchmark_cnn(
        num_layers,
        num_features,
        kernel_size,
        pooling_size,
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

    numdev = len(devlist)

    if data_dir=='':
        gen_data=True
        num_channels=1
        num_classes=2
    else:
        gen_data=False
        imgsize=32
        num_channels=3
        num_classes=10


    # Generate the dataset and build the queue
    if gen_data==True:
        train_data, test_data = build_datasetv2.build_dataset(
                num_trainimg,
                num_testimg,
                imgsize)
    elif gen_data==False:
        train_data, test_data = build_datasetv2.import_cifar(data_dir)

    train_data = train_data.repeat()
    train_data = train_data.shuffle(10*batchsize)
    train_batch = train_data.batch(batchsize)
    iterator = train_batch.make_one_shot_iterator()
    next_batch = iterator.get_next()


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
    tower_predict = []
    tower_labels = []
    tower_loss = []

    for dev_ind in range(numdev):
        dev = devlist[dev_ind]
        print("device %s" % dev)
        with tf.device(devlist[dev_ind]):
            with tf.name_scope('tower_%d' % (dev_ind)) as scope:
                images,labels = next_batch
                loss, logits = cnn_multidevicev2.build_model(
                        images,
                        labels,
                        num_layers,
                        num_features,
                        kernel_size,
                        pooling_size,
                        fully_connected_size,
                        num_channels,
                        num_classes)
                tf.get_variable_scope().reuse_variables()

                gradient = optimizer.compute_gradients(loss)
                tower_gradients.append(gradient)
                tower_predict.append(tf.argmax(logits,1))
                tower_labels.append(tf.argmax(labels,1))

    tower_predict = tf.stack(tower_predict)
    tower_labels = tf.stack(tower_labels)
    mean_gradient = average_gradients.average_gradients(tower_gradients)
    train_op = optimizer.apply_gradients(mean_gradient, global_step=global_step)
    correct_prediction = tf.equal(tower_predict, tower_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss_summary = tf.summary.scalar("training_loss", loss)
    accuracy_summary = tf.summary.scalar("training_accuracy", accuracy)
    lr_summary = tf.summary.scalar("learning_rate", lr)


    acc = np.empty([numsteps,1])
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        writer = tf.summary.FileWriter(train_dir, sess.graph, flush_secs=60)
        t_train = time.time()
        for i in range(numsteps):
            _, loss_summ, acc_summ, lr_summ = sess.run([
                            train_op,
                            loss_summary,
                            accuracy_summary,
                            lr_summary],
                    options=options,
                    run_metadata=run_metadata)
            writer.add_summary(loss_summ, i)
            writer.add_summary(acc_summ, i)
            writer.add_summary(lr_summ, i)
            if logstep > 0:
                if i%logstep==0:
                    print("%.2f sec, step %d" %(time.time()-t_train, i))

                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('%s/timeline_step_%d.json' % (train_dir,i), 'w') as f:
                        f.write(chrome_trace)

        timeUsed_train = time.time()-t_train

        t_infer = time.time()
        acc_validation = sess.run(accuracy)
        timeUsed_infer = time.time() - t_infer
        print("After %d steps: accuracy = %.2f" %(numsteps, acc_validation))

    return timeUsed_train, timeUsed_infer
