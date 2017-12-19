import tensorflow as tf
from utilsTF.benchmarkOPs import Benchmark_MatMul, Benchmark_Conv

tf.app.flags.DEFINE_integer('matsize', 1024, 'size of each matrix for benchmark')
tf.app.flags.DEFINE_integer('kernelsize', 15, 'size of kernel for benchmarking convolution')
tf.app.flags.DEFINE_integer('iter', 10, 'Number opf iterations')
tf.app.flags.DEFINE_string('dev', '/gpu:0', 'Device')
tf.app.flags.DEFINE_bool('testMatMul', True, ' Benchmark matrix multiplication')
tf.app.flags.DEFINE_bool('testConv', True, ' Benchmark 2D convolution')
tf.app.flags.DEFINE_bool

FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.testMatMul:
        ops = FLAGS.matsize**3 + (FLAGS.matsize-1)*FLAGS.matsize**2
        timeUsed = Benchmark_MatMul(FLAGS.matsize,FLAGS.iter,FLAGS.dev)
        print("%d x %d matrix multiplication: %.2f GFLOPS (%.2f matrices per sec)" % (FLAGS.matsize,FLAGS.matsize,ops*1e-9/timeUsed,1/timeUsed))
    if FLAGS.testConv:
        ops = FLAGS.matsize**2 * (FLAGS.kernelsize**3 + (FLAGS.kernelsize-1)*FLAGS.kernelsize**2)
        timeUsed = Benchmark_Conv(FLAGS.matsize,FLAGS.kernelsize,FLAGS.iter,FLAGS.dev)
        print("%d x %d convolution: %.2f GFLOPS (%.2f matrices per sec)" % (FLAGS.matsize,FLAGS.kernelsize,ops*1e-9/timeUsed,1/timeUsed))

if __name__ == '__main__':
  tf.app.run()
