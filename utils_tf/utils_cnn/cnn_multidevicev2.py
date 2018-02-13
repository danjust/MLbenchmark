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


    for layer_ind in range(num_layers):
        if layer_ind==0:
            input_use = images
            input_features = num_channels
        else:
            input_use = pool
            input_features = num_features[layer_ind-1]

        with tf.variable_scope('conv_%d' %layer_ind) as scope:
            kernel = tf.get_variable(
                    'weights_%d' %layer_ind,
                    shape=[kernel_size[0], kernel_size[0], input_features, num_features[layer_ind]],
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

            conv_nonlinear = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        pool = tf.nn.max_pool(
                value=conv_nonlinear,
                ksize=[1, pooling_size[0], pooling_size[0], 1],
                strides=[1, pooling_size[0], pooling_size[0], 1],
                padding='SAME',
                name='pool_%d' %layer_ind)


    pool_flat = tf.reshape(
            tensor=pool,
            shape=[-1, (pool.shape[1]*pool.shape[2]*pool.shape[3]).value])


    dim = pool_flat.get_shape()[1].value
    with tf.variable_scope('dense') as scope:
        weights = tf.get_variable(
                'weights_dense',
                shape=[dim, fully_connected_size],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biases = tf.get_variable(
                'biases_dense',
                shape=[fully_connected_size],
                initializer=tf.constant_initializer(0.1))
        dense = tf.nn.relu(tf.matmul(pool_flat, weights) + biases, name=scope.name)


    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=True)

    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable(
                'weights_logits',
                shape=[fully_connected_size, num_classes],
                initializer=tf.truncated_normal_initializer(stddev=5e-3, dtype=tf.float32))
        biases = tf.get_variable(
                'biases_logits',
                shape=[num_classes],
                initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(dropout, weights), biases, name=scope.name)


    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits)

    return loss, logits
