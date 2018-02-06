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
        lr_initial,
        lr_decay,
        fully_connected_size,
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

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, num_channels])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])


        tower_gradients = []
        tower_predict = []
        tower_labels = []
        tower_loss = []

        with tf.device("/cpu:0"):
            inputs_split = tf.split(x,numdev,axis=0)
            labels_split = tf.split(y_,numdev,axis=0)

        for dev_ind in range(numdev):
            dev = devlist[dev_ind]
            print("device %s" % dev)
            with tf.device(devlist[dev_ind]):
                with tf.name_scope('tower_%d' % (dev_ind)):#,reuse=(dev_ind>0)):
                    input_tower = inputs_split[dev_ind]
                    labels_tower = labels_split[dev_ind]
                    for layer_ind in range(num_layers):
                        if layer_ind==0:
                            input_use = input_tower
                            input_features = num_channels
                        else:
                            input_use = pool
                            input_features = num_features[layer_ind-1]

                        kernel = tf.get_variable(
                                'weights_%d' %layer_ind,
                                shape=[conv_kernel[0], conv_kernel[0], input_features, num_features[layer_ind]],
                                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                        biases = tf.get_variable(
                                'biases_%d' %layer_ind,
                                shape=[num_features[layer_ind]],
                                initializer=tf.constant_initializer(0.0))
                        conv = tf.nn.conv2d(
                                input=input_use,
                                filter=kernel,
                                strides=[1,1,1,1],
                                padding='SAME')

                        conv_nonlinear = tf.nn.relu(tf.nn.bias_add(conv, biases))

                        pool = tf.nn.max_pool(
                                value=conv_nonlinear,
                                ksize=[1, pooling[0], pooling[0], 1],
                                strides=[1, pooling[0], pooling[0], 1],
                                padding='SAME',
                                name='pool_%d' %layer_ind,)


                    pool_flat = tf.reshape(
                            tensor=pool,
                            shape=[-1, (pool.shape[1]*pool.shape[2]*pool.shape[3]).value])


                    dim = pool_flat.get_shape()[1].value
                    weights = tf.get_variable(
                            'weights_dense',
                            shape=[dim, fully_connected_size],
                            initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                    biases = tf.get_variable(
                            'biases_dense',
                            shape=[fully_connected_size],
                            initializer=tf.constant_initializer(0.1))
                    dense = tf.nn.relu(tf.matmul(pool_flat, weights) + biases)


                    dropout = tf.layers.dropout(
                            inputs=dense,
                            rate=0.4,
                            training=True)


                    weights = tf.get_variable(
                            'weights_logits',
                            shape=[fully_connected_size, num_classes],
                            initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
                    biases = tf.get_variable(
                            'biases_logits',
                            shape=[num_classes],
                            initializer=tf.constant_initializer(0.1))
                    logits = tf.matmul(dropout, weights) + biases


                    loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=labels_tower,
                            logits=logits)

                    tf.get_variable_scope().reuse_variables()


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
