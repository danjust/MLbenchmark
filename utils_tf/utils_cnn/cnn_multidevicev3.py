"""Function to build the graph of a CNN with flexible hyperparameters
    - number of layers
    - size of the convolution kernel
    - pooling size
"""

import tensorflow as tf

def build_model(
        images,
        labels,
        num_layers,
        num_features,
        kernel_size,
        pooling_size,
        fully_connected_size,
        num_channels,
        num_classes):

    # Conv layer 0
    with tf.variable_scope('conv_0') as scope:
        kernel0 = tf.get_variable(
                'weights_0',
                shape=[kernel_size[0], kernel_size[0], num_channels, num_features[0]],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biases0 = tf.get_variable(
                'biases_0',
                shape=[num_features[0]],
                initializer=tf.constant_initializer(0.0))
        conv0 = tf.nn.conv2d(
                input=images,
                filter=kernel0,
                strides=[1,1,1,1],
                padding='SAME')

        conv_nonlinear0 = tf.nn.relu(tf.nn.bias_add(conv0, biases0), name=scope.name)

    pool0 = tf.nn.max_pool(
            value=conv_nonlinear0,
            ksize=[1, pooling_size[0], pooling_size[0], 1],
            strides=[1, pooling_size[0], pooling_size[0], 1],
            padding='SAME',
            name='pool_0')


    # Conv layer 1
    with tf.variable_scope('conv_1') as scope:
        kernel1 = tf.get_variable(
                'weights_1',
                shape=[kernel_size[1], kernel_size[1], num_features[0], num_features[1]],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biases1 = tf.get_variable(
                'biases_1',
                shape=[num_features[1]],
                initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
                input=pool1,
                filter=kernel1,
                strides=[1,1,1,1],
                padding='SAME')

        conv_nonlinear1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name=scope.name)

    pool1 = tf.nn.max_pool(
            value=conv_nonlinear1,
            ksize=[1, pooling_size[1], pooling_size[1], 1],
            strides=[1, pooling_size[1], pooling_size[1], 1],
            padding='SAME',
            name='pool_1')


    # Conv layer 2
    with tf.variable_scope('conv_2') as scope:
        kernel2 = tf.get_variable(
                'weights_2',
                shape=[kernel_size[2], kernel_size[2], num_features[1], num_features[2]],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biases2 = tf.get_variable(
                'biases_2',
                shape=[num_features[2]],
                initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
                input=pool2,
                filter=kernel2,
                strides=[1,1,1,1],
                padding='SAME')

        conv_nonlinear2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name=scope.name)

    pool2 = tf.nn.max_pool(
            value=conv_nonlinear2,
            ksize=[1, pooling_size[2], pooling_size[2], 1],
            strides=[1, pooling_size[2], pooling_size[2], 1],
            padding='SAME',
            name='pool_2')


    pool_flat = tf.reshape(
            tensor=pool2,
            shape=[-1, (pool2.shape[1]*pool2.shape[2]*pool2.shape[3]).value])


    dim = pool_flat.get_shape()[1].value
    with tf.variable_scope('dense') as scope:
        weightsd = tf.get_variable(
                'weights_dense',
                shape=[dim, fully_connected_size],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biasesd = tf.get_variable(
                'biases_dense',
                shape=[fully_connected_size],
                initializer=tf.constant_initializer(0.1))
        dense = tf.nn.relu(tf.matmul(pool_flat, weightsd) + biasesd, name=scope.name)


    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=True)

    with tf.variable_scope('softmax') as scope:
        weightss = tf.get_variable(
                'weights_logits',
                shape=[fully_connected_size, num_classes],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biasess = tf.get_variable(
                'biases_logits',
                shape=[num_classes],
                initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(dropout, weightss), biasess, name=scope.name)


    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits)

    return loss, logits
