import tensorflow as tf
import time
from tensorflow.python.client import timeline
from utils_tf.utils_data import build_dataset

def feed_dict_from_memory(
        batchsize,
        data_file,
        devlist,
        num_trainimg,
        imgsize,
        numsteps,
        log_dir,
        logstep,
        datatype):

    numdev = len(devlist)

    if data_file=='':
        gen_data=True
        num_channels=1
        num_classes=2
    else:
        gen_data=False
        imgsize=32
        num_channels=3
        num_classes=10

    total_batchsize = int(batchsize*len(devlist))

    # Generate or load data
    with tf.device("/cpu:0"):
        if gen_data==True:
            trainimg, trainlabel, _, _ = build_dataset.generate_data(
                    num_trainimg,
                    0,
                    imgsize,
                    datatype)
        elif gen_data==False:
            trainimg, trainlabel = build_dataset.load_full_dataset(
                    data_file,
                    imgsize,
                    imgsize)

        x = tf.placeholder(
                tf.float32,
                shape=[None, imgsize, imgsize, num_channels])
        y_ = tf.placeholder(
                tf.float32,
                shape=[None, num_classes])

        # split batches
        inputs_split = tf.split(x,numdev,axis=0)
        labels_split = tf.split(y_,numdev,axis=0)

    # Generate graph
    returnValue = []
    for dev_ind in range(numdev):
        dev = devlist[dev_ind]
        print("device %s" % dev)
        with tf.device(devlist[dev_ind]):
            with tf.name_scope('tower_%d' % (dev_ind)):#,reuse=(dev_ind>0)):
                input_tower = inputs_split[dev_ind]
                labels_tower = labels_split[dev_ind]
                returnValue.append(input_tower[0,0,0,0])

    # Run model
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
                    print("%.2f sec, step %d" %(time.time()-t_start, i))
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('%s/timeline_step_%d.json' % (log_dir,i), 'w') as f:
                        f.write(chrome_trace)
        timeUsed = time.time()-t_start

    return timeUsed
