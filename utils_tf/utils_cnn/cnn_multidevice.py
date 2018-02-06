"""Function to build the graph of a CNN with flexible hyperparameters
    - number of layers
    - size of the convolution kernel
    - pooling size
"""

import tensorflow as tf
from utils_tf.utils_cnn import average_gradients


def build_graph(
        imgsize,
        num_channels,
        num_classes,
        batchsize,
        devlist):

    numdev = len(devlist)
    print(numdev)
    g = tf.Graph()

    with g.as_default():
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, num_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])


        tower_gradients = []
        tower_predict = []
        tower_labels = []
        tower_loss = []

        batchsize_per_gpu = batchsize/numdev
        with tf.device("/cpu:0"):
            inputs_split = tf.split(x,numdev,axis=0)
            labels_split = tf.split(y_,numdev,axis=0)

        for dev_ind in range(numdev):
            dev = devlist[dev_ind]
            print("device %s" % dev)
            print(dev_ind)
            with tf.device('/cpu:0'):#devlist[dev_ind]):
                with tf.variable_scope('tower_%d' %0,reuse=(dev_ind>0)):
                    input_tower = inputs_split[dev_ind]
                    labels_tower = labels_split[dev_ind]

                    with tf.device('/cpu:0'):
                        kernel0 = tf.get_variable(
                                'weights0',
                                shape=[5, 5, num_channels, 8],
                                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                        biases0 = tf.get_variable(
                                'biases0',
                                shape=[8],
                                initializer=tf.constant_initializer(0.0))
                    conv0 = tf.nn.conv2d(
                            input=input_tower,
                            filter=kernel0,
                            strides=[1,1,1,1],
                            padding='SAME')

                    conv_nonlinear0 = tf.nn.relu(tf.nn.bias_add(conv0, biases0))

                    pool0 = tf.nn.max_pool(
                            value=conv_nonlinear0,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool0')


                    kernel1 = tf.get_variable(
                            'weights1',
                            shape=[5, 5, 8, 64],
                            initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                    biases1 = tf.get_variable(
                            'biases1',
                            shape=[64],
                            initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(
                            input=pool0,
                            filter=kernel1,
                            strides=[1,1,1,1],
                            padding='SAME')

                    conv_nonlinear1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1))

                    pool1 = tf.nn.max_pool(
                            value=conv_nonlinear1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool1')


                    pool_flat = tf.reshape(
                            tensor=pool1,
                            shape=[-1, (pool1.shape[1]*pool1.shape[2]*pool1.shape[3]).value])


                    dim = pool_flat.get_shape()[1].value
                    with tf.device('/cpu:0'):
                        weights = tf.get_variable(
                                'weights_dense',
                                shape=[dim, 256],
                                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                        biases = tf.get_variable(
                                'biases_dense',
                                shape=[256],
                                initializer=tf.constant_initializer(0.1))
                    dense = tf.nn.relu(tf.matmul(pool_flat, weights) + biases)


                    dropout = tf.layers.dropout(
                            inputs=dense,
                            rate=0.4,
                            training=True)


                    with tf.device('/cpu:0'):
                        weights = tf.get_variable(
                                'weights_logits',
                                shape=[256, num_classes],
                                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                        biases = tf.get_variable(
                                'biases_logits',
                                shape=[num_classes],
                                initializer=tf.constant_initializer(0.1))
                    logits = tf.matmul(dropout, weights) + biases


                    loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=labels_tower,
                            logits=logits)


                    gradient = optimizer.compute_gradients(loss)
                    tower_gradients.append(gradient)
                    tower_predict.append(tf.argmax(logits,1))
                    tower_labels.append(tf.argmax(labels_split[dev_ind],1))

        tower_predict = tf.stack(tower_predict)
        tower_labels = tf.stack(tower_labels)
        mean_gradient = average_gradients.average_gradients(tower_gradients)
        train_op = optimizer.apply_gradients(mean_gradient, global_step=global_step)
        correct_prediction = tf.equal(tower_predict, tower_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return g, x, y_, train_op, loss, accuracy
