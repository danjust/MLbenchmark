"""Benchmark script for frequently used machine learning operations
Using TensorFlow
"""
import os
import argparse
import tensorflow as tf
from  utils_tf import benchmark_matmul
from  utils_tf import benchmark_conv
from  utils_tf import benchmark_rnn
from  utils_tf import benchmark_cnnv2
from  utils_tf import benchmark_latency
from  utils_tf import benchmark_inputpipeline
from  utils_tf import benchmark_connectivity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


parser = argparse.ArgumentParser('Benchmarking different aspects of a machine learning algorithm')

# Benchmarks to perform
parser.add_argument('--testMatMul', action="store_true", default=False, help='Benchmark matrix multiplication')
parser.add_argument('--testConv', action="store_true", default=False, help='Benchmark 2D convolution')
parser.add_argument('--testRNN', action="store_true", default=False, help='Benchmark recurrent neural networks')
parser.add_argument('--testCNN', action="store_true", default=False, help='Benchmark a cnn training')
parser.add_argument('--testLatency', action="store_true", default=False, help='Benchmark the latency of a Device')
parser.add_argument('--testPipeline', action="store_true", default=False, help='Benchmark the data pipeline')
parser.add_argument('--testConnectivity', action="store_true", default=False, help='Benchmark the conncetion speed of a list of devices')

# General parameters
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--devlist', type=str, default='', help='List of devices to use, overwrites num_gpu if set')
parser.add_argument('--precision', type=int, default=32, help='Precision')
parser.add_argument('--keep_in_mem', action="store_true", default=False, help='Keep all data in memory, only if data_path is set')

# Parameters for matrix multiplication / convolution
parser.add_argument('--iter', type=int, default=10, help='Number of iterations')
parser.add_argument('--logFLOPs', type=int, default=0, help='log10 of number of FLOPs to perform (will be rounded up), overwrites iter, set 0 to use iter')
parser.add_argument('--matsize', type=int, default=1024, help='Size of each matrix for benchmark')
parser.add_argument('--kernelsize', type=int, default=15, help='Size of kernel for benchmarking convolution')
parser.add_argument('--lr_initial', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')

# Parameters for RNNs
parser.add_argument('--rnn_type', type=str, default='rnn', help='Type of RNN (rnn or lstm)')
parser.add_argument('--seq_length', type=int, default=50, help='Length of sequence')
parser.add_argument('--batch_size_rnn', type=int, default=64, help='Batch size')
parser.add_argument('--num_samples', type=int, default=10000, help='Total number of samples of length seq_length')
parser.add_argument('--num_units', type=int, default=32, help='Number of hidden units')
parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
parser.add_argument('--iter_rnn', type=int, default=10, help='Number of iterations for RNNs')

# Parameters for CNNs
parser.add_argument('--data_file', type=str, default='', help='.tfrecords file with image data, leave empty for synthetic data')
parser.add_argument('--train_dir', type=str, default='/tmp/train', help='directory for logging')
parser.add_argument('--num_layers_cnn', type=int, default=3, help='Number of convolution/pooling layers in CNN')
parser.add_argument('--num_features', type=int, nargs='+', default=[16,64,128], help='Vector containing the number of features in each convolutional layer')
parser.add_argument('--kernel_cnn', type=int, nargs='+', default=[5,3,3], help='Vector containing the kernelsize in each convolutional layer')
parser.add_argument('--pooling_cnn', type=int, nargs='+', default=[2,2,2], help='Vector containing the size of max pooling in each pooling layer')
parser.add_argument('--fully_connected_size', type=int, default=256, help='Number of neurons in fully connected layer')
parser.add_argument('--num_trainimg', type=int, default=10000, help='Number of training images if synthetic data')
parser.add_argument('--num_testimg', type=int, default=10000, help='Number of validation images if synthetic data')
parser.add_argument('--logstep', type=int, default=500, help='write log at these steps (0 to disable logging)')
parser.add_argument('--trackingstep', type=int, default=5000, help='write tracking at these steps (0 to disable logging)')
parser.add_argument('--imgsize', type=int, default=50, help='Size of (square) images')
parser.add_argument('--numsteps_cnn', type=int, default=10000, help='Number of steps to train CNN')
parser.add_argument('--batchsize_cnn', type=int, default=128, help='Batch size for training CNN')


parser.add_argument('--iterations_latency', type=int, default=100000, help='Number of iterations for latency')
parser.add_argument('--device1_latency', type=str, default='/cpu:0', help='First device for latency test')
parser.add_argument('--device2_latency', type=str, default='/gpu:0', help='Second device for latency test')

parser.add_argument('--pipeline', type=str, default='dataset', help='Type of data pipeline, one of feed_dict, queue_runner, dataset')
parser.add_argument('--numsteps_input', type=int, default=10000, help='Number of iterations for testing data pipeline')
parser.add_argument('--batchsize_input', type=int, default=128, help='Batch size for testing data pipeline')

parser.add_argument('--devlist_connectivity', type=str, default='/cpu:0,/gpu:0', help='Devices for testing connection speed')
parser.add_argument('--matsize_connectivity', type=int, default=2048, help='Size of tensors for testing connection speed')
parser.add_argument('--iterations_connectivity', type=int, default=1000, help='Number of iterations for testing connection speed')

args = parser.parse_args()

def main(_):
    if args.testMatMul:
        ops = (args.matsize**3
                + (args.matsize-1)*args.matsize**2)
                # matsize**3 multiplications,
                # (matsize-1)*matsize**2 additions
        print("========================================\n")
        print("Start matrix multiplication")
        timeUsed = benchmark_matmul.benchmark_matmul(
                args.matsize,
                args.iter,
                args.logFLOPs,
                args.num_gpu,
                args.devlist,
                args.precision)
        print("\n%d x %d matrix multiplication (float%d): %.2f GFLOPS (%.2f matrices per sec)"
                % (args.matsize,
                args.matsize,
                args.precision,
                ops*1e-9/timeUsed,
                1/timeUsed))

    if args.testConv:
        ops = ((args.matsize-args.kernelsize+1)**2
                * (args.kernelsize**3
                + (args.kernelsize-1)*args.kernelsize**2))
                # (matsize.kernelsize+1)**2 GEMMs
        print("========================================\n")
        print("Start convolution")
        timeUsed = benchmark_conv.benchmark_conv(
                args.matsize,
                args.kernelsize,
                args.iter,
                args.num_gpu,
                args.devlist,
                args.precision)
        print("\n%d x %d convolution (float%d): %.2f GFLOPS (%.2f matrices per sec)"
                % (args.matsize,
                args.kernelsize,
                args.precision,
                ops*1e-9/timeUsed,
                1/timeUsed))

    if args.testRNN:
        print("========================================\n")
        print("Start recurrent neural network (%s)" %args.rnn_type)
        timeUsed = benchmark_rnn.benchmark_rnn(
                args.rnn_type,
                args.seq_length,
                args.batch_size_rnn,
                args.num_samples,
                args.num_units,
                args.num_classes,
                args.lr_initial,
                args.iter_rnn,
                args.num_gpu,
                args.devlist,
                args.precision)
        print("\n%s:  %.2f steps per sec"
                % (args.rnn_type,
                1/timeUsed))

    if args.testCNN:
        print("========================================\n")
        print("Start training convolutional neural network")
        timeUsed_train, timeUsed_infer = benchmark_cnnv2.benchmark_cnn(
                args.num_layers_cnn,
                args.num_features,
                args.kernel_cnn,
                args.pooling_cnn,
                args.fully_connected_size,
                args.lr_initial,
                args.lr_decay,
                args.precision,
                args.num_trainimg,
                args.num_testimg,
                args.imgsize,
                args.numsteps_cnn,
                args.batchsize_cnn,
                args.keep_in_mem,
                args.logstep,
                args.trackingstep,
                args.num_gpu,
                args.devlist,
                args.data_file,
                args.train_dir)
        print("========================================\n")
        numdev=max(1,args.num_gpu)
        print("convolutional neural network, %d training steps: " \
                "%.2f steps per sec (%2.f images per sec) \n" \
                "%d images inferred: %.2f sec (%.2f images per sec)"
                % (args.numsteps_cnn,
                args.numsteps_cnn/timeUsed_train,
                args.numsteps_cnn*args.batchsize_cnn*numdev/timeUsed_train,
                args.num_testimg,
                timeUsed_infer,
                args.num_testimg/timeUsed_infer
                ))

    if args.testLatency:
        print("========================================\n")
        print("Start testing GPU latency")
        timeUsed = benchmark_latency.benchmark_latency(
                args.iterations_latency,
                args.device1_latency,
                args.device2_latency)
        print("\nAverage latency = %f ms" % (timeUsed*1000))

    if args.testPipeline:
        print("========================================\n")
        print("Start testing input pipeline")
        timeUsed = benchmark_inputpipeline.benchmark_pipeline(
                args.num_gpu,
                args.devlist,
                args.batchsize_input,
                args.data_file,
                args.keep_in_mem,
                args.num_trainimg,
                args.imgsize,
                args.precision,
                args.numsteps_input,
                args.pipeline,
                args.train_dir,
                args.logstep)
        num_imgs = args.numsteps_input*args.batchsize_input*max(1,args.num_gpu)
        print("\n%.2f images per second" %(num_imgs/timeUsed))

    if args.testConnectivity:
        print("========================================\n")
        print("Start testing input pipeline")
        benchmark_connectivity.benchmark_connectivity(
                args.devlist_connectivity,
                args.precision,
                args.matsize_connectivity,
                args.matsize_connectivity,
                args.iterations_connectivity)
if __name__ == '__main__':
  tf.app.run()
