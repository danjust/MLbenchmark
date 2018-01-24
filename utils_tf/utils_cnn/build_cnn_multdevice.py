"""Function to build the graph of a CNN with flexible hyperparameters
    - number of layers
    - size of the convolution kernel
    - pooling size
"""

import tensorflow as tf
from utils_tf.utils_cnn import average_gradients

def build_graph(
        num_layers,
        num_features,
        conv_kernel,
        pooling,
        imgsize,
        devlist):

    numdev = len(devlist)
    g = tf.Graph()

    with g.as_default():
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        batchsize = tf.shape(x)[0]

        tower_gradients = []
        tower_predict = []
        tower_labels = []

        batchsize_per_gpu = tf.cast(batchsize/numdev,tf.int32)


        for i in range(numdev):
            dev = devlist[i]
            print("device %s" % dev)
            with tf.device(dev):
                with tf.name_scope('tower_%d' % i) as scope:
                    inputs = x[i*batchsize_per_gpu:(i+1)*batchsize_per_gpu,:,:,:]
                    labels = y_[i*batchsize_per_gpu:(i+1)*batchsize_per_gpu,:]

                    for i in range(num_layers):
                        conv = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_features[i],
                                kernel_size=conv_kernel[i],
                                strides=[1,1],
                                padding='SAME',
                                activation=tf.nn.relu)

                        pool = tf.layers.max_pooling2d(
                                inputs=conv,
                                pool_size=pooling[i],
                                strides=pooling[i],
                                padding='VALID')

                        inputs = pool


                    pool_flat = tf.reshape(
                            tensor=pool,
                            shape=[-1, (pool.shape[1]*pool.shape[2]*pool.shape[3]).value])


                    dense = tf.layers.dense(
                            inputs=pool_flat,
                            units=1024,
                            activation=tf.nn.relu)


                    dropout = tf.layers.dropout(
                            inputs=dense,
                            rate=0.4,
                            training=True)


                    logits = tf.layers.dense(inputs=dropout, units=2)

                    loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=labels, logits=logits)

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

    return g, x, y_, train_op, loss, accuracy
