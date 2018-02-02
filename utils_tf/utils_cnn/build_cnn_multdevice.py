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
        num_channels,
        num_classes,
        devlist):

    numdev = len(devlist)
    print(numdev)
    g = tf.Graph()

    with g.as_default():
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)

        lr = tf.train.exponential_decay(.0001,
                                    global_step,
                                    10000,
                                    .00001,
                                    staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, num_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

        batchsize = tf.shape(x)[0]

        tower_gradients = []
        tower_predict = []
        tower_labels = []
        tower_loss = []

        batchsize_per_gpu = tf.cast(batchsize/numdev,tf.int32)
        with tf.device("/cpu:0"):
            inputs_split = tf.split(x,numdev,axis=0)
            labels_split = tf.split(y_,numdev,axis=0)

        for dev_ind in range(numdev):
            dev = devlist[dev_ind]
            print("device %s" % dev)
            print(dev_ind)
            with tf.device(devlist[dev_ind]):
                with tf.variable_scope('tower_%d' %dev_ind) as scope:
                    input_tower = inputs_split[dev_ind]
                    labels_tower = labels_split[dev_ind]
                    for layer_ind in range(num_layers):
                        if layer_ind==0:
                            inputs = input_tower
                        else:
                            inputs = pool

                        conv = tf.layers.conv2d(
                                inputs=inputs,
                                filters=num_features[layer_ind],
                                kernel_size=conv_kernel[layer_ind],
                                strides=[1,1],
                                padding='SAME',
                                activation=tf.nn.relu)

                        pool = tf.layers.max_pooling2d(
                                inputs=conv,
                                pool_size=pooling[layer_ind],
                                strides=pooling[layer_ind],
                                padding='VALID')


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


                    logits = tf.layers.dense(inputs=dropout, units=num_classes)

                    loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=labels_tower,
                            logits=logits)

                    params = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope='tower_%d' %dev_ind)

                    tf.get_variable_scope().reuse_variables()

                    gradient = optimizer.compute_gradients(loss,var_list = params)
                    tower_gradients.append(gradient)
                    tower_predict.append(tf.argmax(logits,1))
                    tower_labels.append(tf.argmax(labels_split[dev_ind],1))

        tower_predict = tf.stack(tower_predict)
        tower_labels = tf.stack(tower_labels)
        mean_gradient = average_gradients.average_gradients(tower_gradients)
        train_op = optimizer.apply_gradients(mean_gradient, global_step=global_step)
        correct_prediction = tf.equal(tower_predict, tower_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return g, x, y_, train_op, loss, accuracy, tower_predict
