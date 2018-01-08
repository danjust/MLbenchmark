"""Benchmark script for frequently used machine learning operations
Using TensorFlow
"""

import tensorflow as tf
from  utils_tf import benchmark_matmul
from  utils_tf import benchmark_conv
from  utils_tf import benchmark_rnn
from  utils_tf import benchmark_cnn


# Benchmarks to perform
tf.app.flags.DEFINE_bool('testMatMul', True, 'Benchmark matrix multiplication')
tf.app.flags.DEFINE_bool('testConv', True, 'Benchmark 2D convolution')
tf.app.flags.DEFINE_bool('testRNN', False, 'Benchmark recurrent neural networks')
tf.app.flags.DEFINE_bool('testCNN', False, 'Benchmark a cnn training on cifar10')

# General parameters
tf.app.flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs to use')
tf.app.flags.DEFINE_string('devlist', '', 'List of devices to use, overwrites num_gpu if set')
tf.app.flags.DEFINE_string('datatype', 'float32', 'Datatype')

# Parameters for matrix multiplication / convolution
tf.app.flags.DEFINE_integer('iter', 10, 'Number of iterations')
tf.app.flags.DEFINE_integer('matsize', 1024, 'Size of each matrix for benchmark')
tf.app.flags.DEFINE_integer('kernelsize', 15, 'Size of kernel for benchmarking convolution')

# Parameters for RNNs
tf.app.flags.DEFINE_string('rnn_type', 'rnn', 'Type of RNN (rnn or lstm)')
tf.app.flags.DEFINE_integer('seq_length', 50, 'Length of sequence')
tf.app.flags.DEFINE_integer('batch_size_rnn', 64, 'Batch size')
tf.app.flags.DEFINE_integer('num_samples', 10000, 'Total number of samples of length seq_length')
tf.app.flags.DEFINE_integer('num_units', 32, 'Number of hidden units')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of target classes')
tf.app.flags.DEFINE_integer('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_integer('iter_rnn', 10, 'Number of iterations for RNNs')

# Parameters for CNNs
tf.app.flags.DEFINE_integer('steps_cnn', 100, 'Number of steps to train CNN')
tf.app.flags.DEFINE_integer('batch_size_cnn', 128, 'Number of steps to train CNN')
tf.app.flags.DEFINE_string('data_dir', '/data', 'Directory with training data')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.testMatMul:
        ops = (FLAGS.matsize**3
                + (FLAGS.matsize-1)*FLAGS.matsize**2)
                # matsize**3 multiplications,
                # (matsize-1)*matsize**2 additions
        timeUsed = benchmark_matmul.benchmark_matmul(
                FLAGS.matsize,
                FLAGS.iter,
                FLAGS.num_gpu,
                FLAGS.devlist,
                FLAGS.datatype)
        print("\n%d x %d matrix multiplication (%s): %.2f GFLOPS (%.2f matrices per sec)"
                % (FLAGS.matsize,
                FLAGS.matsize,
                FLAGS.datatype,
                ops*1e-9/timeUsed,
                1/timeUsed))

    if FLAGS.testConv:
        ops = ((FLAGS.matsize-FLAGS.kernelsize+1)**2
                * (FLAGS.kernelsize**3
                + (FLAGS.kernelsize-1)*FLAGS.kernelsize**2))
                # (matsize.kernelsize+1)**2 GEMMs
        timeUsed = benchmark_conv.benchmark_conv(
                FLAGS.matsize,
                FLAGS.kernelsize,
                FLAGS.iter,
                FLAGS.num_gpu,
                FLAGS.devlist,
                FLAGS.datatype)
        print("\n%d x %d convolution (%s): %.2f GFLOPS (%.2f matrices per sec)"
                % (FLAGS.matsize,
                FLAGS.kernelsize,
                FLAGS.datatype,
                ops*1e-9/timeUsed,
                1/timeUsed))

    if FLAGS.testRNN:
        timeUsed = benchmark_rnn.benchmark_rnn(
                FLAGS.rnn_type,
                FLAGS.seq_length,
                FLAGS.batch_size_rnn,
                FLAGS.num_samples,
                FLAGS.num_units,
                FLAGS.num_classes,
                FLAGS.learning_rate,
                FLAGS.iter_rnn,
                FLAGS.num_gpu,
                FLAGS.devlist,
                FLAGS.datatype)
        print("\n%s:  %.2f steps per sec"
                % (FLAGS.rnn_type,
                1/timeUsed))

    if FLAGS.testCNN:
        timeUsed = benchmark_cnn.train(
                FLAGS.data_dir,
                FLAGS.batch_size_cnn,
                FLAGS.steps_cnn,
                FLAGS.num_gpu,
                FLAGS.devlist)
        print("convolutional neural network, %d training steps: %.2f steps per sec (%2.f images per sec)"
                % (FLAGS.steps_cnn,
                FLAGS.steps_cnn/timeUsed,
                FLAGS.steps_cnn*FLAGS.batch_size_cnn/timeUsed))

if __name__ == '__main__':
  tf.app.run()
