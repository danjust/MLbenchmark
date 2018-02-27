import tensorflow as tf
import time
from tensorflow.python.client import timeline
from utils_tf.utils_data import build_dataset

def data_api_from_memory(
        batchsize,
        data_file,
        devlist,
        num_trainimg,
        imgsize,
        numsteps,
        log_dir,
        logstep,
        datatype):


    with tf.device('/cpu:0'):

        numdev = len(devlist)

        if data_file=='':
            gen_data = True
            data_in_mem = True
        else:
            gen_data = False
            imgsize = 32

        # Generate or load data
        if gen_data==True:
            trainimg, trainlabel, testimg, testlabel = build_dataset.generate_data(
                    num_trainimg,
                    0,
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


    # Define graph
    returnValue = []
    for dev_ind in range(numdev):
        dev = devlist[dev_ind]
        print("device %s" % dev)
        with tf.device(devlist[dev_ind]):
            with tf.name_scope('tower_%d' % (dev_ind)) as scope:
                images, labels = iterator.get_next()
                returnValue.append(images[0,0,0,0])


    # Run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        t_start = time.time()
        for i in range(numsteps):
            _ = sess.run(returnValue)
            if logstep > 0:
                if i%(logstep)==0:
                    print("%.2f sec, step %d" %(time.time()-t_start, i))
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('%s/timeline_step_%d.json' % (log_dir,i), 'w') as f:
                        f.write(chrome_trace)
        timeUsed = time.time()-t_start

    return timeUsed


def data_api_from_file(
        batchsize,
        data_file,
        devlist,
        imgsize,
        numsteps,
        log_dir,
        logstep,
        datatype):

    with tf.device('/cpu:0'):
        numdev = len(devlist)

        imgsize = 32

        # Generate dataset and iterator
        filenames = tf.placeholder(tf.string, shape=[None])
        iterator = build_dataset.get_iterator(
                filenames,
                batchsize,
                imgsize,
                imgsize,
                num_channels,
                numdev)

    # Define graph
    returnValue = []
    for dev_ind in range(numdev):
        dev = devlist[dev_ind]
        print("device %s" % dev)
        with tf.device(devlist[dev_ind]):
            with tf.name_scope('tower_%d' % (dev_ind)) as scope:
                images, labels = iterator.get_next()
                images = tf.reshape(images, [batchsize, imgsize, imgsize, num_channels])
                returnValue.append(images[0,0,0,0])


    # Run model
    with tf.Session() as sess:
        sess.run(it.initializer, feed_dict={filenames: data_file})
        sess.run(tf.global_variables_initializer())
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        t_start = time.time()
        for i in range(numsteps):
            _ = sess.run(
                    returnValue,
                    options=options,
                    run_metadata=run_metadata)
            if logstep > 0:
                if i%(logstep)==0:
                    print("%.2f sec, step %d" %(time.time()-t_start, i))
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('%s/timeline_step_%d.json' % (log_dir,i), 'w') as f:
                        f.write(chrome_trace)
        timeUsed = time.time()-t_start



    return timeUsed
