# MLbenchmark
Tools for benchmarking ML applications on different hardware.
Currently only supports tensorflow.

Run the demo in the jupyter notebook or run
```bash
python benchmark.py
```
with the following optional arguments:


#### Types of benchmarks
--testMatMul (Whether to benchmark matrix multiplication, default False)<br/>
--testConv (Whether to benchmark 2D convolution, default False)<br/>
--testRNN (Whether to benchmark recurrent neural networks, default False)<br/>
--testCNN (Whether to benchmark a cnn training on sythetic data, default False)<br/>

#### General parameters
--num_gpu (Number of GPUs to use, default 1)<br/>
--devlist (List of devices to use, overwrites num_gpu if set, default '')<br/>
--datatype (Datatype, default float32)<br/>

#### Parameters for matrix multiplication / convolution
--iter (Number of iterations, default 10)<br/>
--matsize (Size of each matrix for benchmark, default 1024)<br/>
--kernelsize (Size of kernel for benchmarking convolution, default 15)<br/>

#### Parameters for RNNs
--rnn_type (Type of RNN (rnn or lstm), default 'rnn')<br/>
--seq_length (Length of sequence, default 50)<br/>
--batch_size_rnn (Batch size, default 64)<br/>
--num_samples (Total number of samples of length seq_length, default 10000)<br/>
--num_units (Number of hidden units, default 32)<br/>
--num_classes (Number of target classes, default 10)<br/>
--learning_rate (Learning rate, default 0.001)<br/>
--iter_rnn (Number of iterations for RNNs, default 10)<br/>

#### Parameters for CNNs
--num_layers_cnn (Number of convolution/pooling layers in CNN, default 3)<br/>
--num_features (Vector containing the number of features in each convolutional layer,
    default [16,64,128])<br/>
--kernel_cnn (Vector containing the kernel size in each convolutional layer,
    default [3,3,3])<br/>
--pooling_cnn (Vector containing the size of max pooling in each pooling layer',
    default [3,3,3])<br/>
--num_trainimg (Number of training images, default 100000)<br/>
--num_testimg (Number of validation images, default 1000)<br/>
--imgsize (Size of (square) images, default 50)<br/>
--numsteps_cnn (Number of steps to train CNN, default 500)<br/>
--batchsize_cnn (Batch size for training CNN, default 32)<br/>
--logstep_cnn (Write log at these steps (0 to disable logging), default 10)<br/>
