import tensorflow as tf
import time
import numpy as np
from tensorflow.python.client import timeline
from utils_tf.utils_cnn import build_datasetv2, build_dataset

def benchmark_pipeline(
        num_gpu,
        devlist,
        batchsize,
        data_dir,
        num_trainimg,
        imgsize,
        precision,
        numsteps,
        pipeline,
        log_dir,
        logstep):

    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    numdev = len(devlist)

    datatype = "float%d" %precision

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
    if pipeline=="feed_dict":
        total_batchsize = int(batchsize*len(devlist))
        if gen_data==True:
            trainimg, trainlabel, _, _ = build_dataset.build_dataset(
                    num_trainimg,
                    0,
                    imgsize)
        elif gen_data==False:
            trainimg, trainlabel, _, _ = build_dataset.import_cifar(data_dir)

        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, num_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

        with tf.device("/cpu:0"):
            inputs_split = tf.split(x,numdev,axis=0)
            labels_split = tf.split(y_,numdev,axis=0)

        returnValue = []
        for dev_ind in range(numdev):
            dev = devlist[dev_ind]
            print("device %s" % dev)
            with tf.device(devlist[dev_ind]):
                with tf.name_scope('tower_%d' % (dev_ind)):#,reuse=(dev_ind>0)):
                    input_tower = inputs_split[dev_ind]
                    labels_tower = labels_split[dev_ind]
                    returnValue.append(input_tower[0,0,0,0])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            t_start = time.time()
            for i in range(numsteps):
                batch = np.random.randint(
                        0,
                        np.size(trainlabel,0),
                        total_batchsize)
                img_batch = trainimg[batch,:,:]
                label_batch = trainlabel[batch,:]


                _ = sess.run(
                        returnValue,
                        feed_dict={x: img_batch, y_: label_batch},
                        options=options,
                        run_metadata=run_metadata)
                if logstep > 0:
                    if i%logstep==0:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('%s/timeline_step_%d.json' % (log_dir,i), 'w') as f:
                            f.write(chrome_trace)
            timeUsed = time.time()-t_start


    elif pipeline=="queue_runner":
        print('ToDo')


    elif pipeline=="dataset":
        with tf.device('/cpu:0'):
            if gen_data==True:
                train_data, test_data = build_datasetv2.build_dataset(
                        num_trainimg,
                        0,
                        imgsize,
                        datatype)
            elif gen_data==False:
                train_data, test_data = build_datasetv2.import_cifar(data_dir)

            train_data = train_data.repeat()
            train_data = train_data.shuffle(5*batchsize)
            train_batch = train_data.batch(batchsize)
            iterator = train_batch.make_one_shot_iterator()
            next_batch = iterator.get_next()

            returnValue = []
            for dev_ind in range(numdev):
                dev = devlist[dev_ind]
                print("device %s" % dev)
                with tf.device(devlist[dev_ind]):
                    with tf.name_scope('tower_%d' % (dev_ind)) as scope:
                        images,labels = next_batch
                        returnValue.append(images[0,0,0,0])



        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            t_start = time.time()
            for i in range(numsteps):
                _ = sess.run(returnValue)
                if logstep > 0:
                    if i%(logstep)==0:
                        print("%.2f sec, step %d" %(time.time()-t_train, i))
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('%s/timeline_step_%d.json' % (log_dir,i), 'w') as f:
                            f.write(chrome_trace)
            timeUsed = time.time()-t_start


    return timeUsed
