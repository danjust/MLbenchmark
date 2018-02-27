import tensorflow as tf
import time
import numpy as np
from tensorflow.python.client import timeline
from utils_tf.utils_connectivity import benchmark_feed_dict, benchmark_queue_runner, benchmark_data_api

def benchmark_pipeline(
        num_gpu,
        devlist,
        batchsize,
        data_file,
        data_in_mem,
        num_trainimg,
        imgsize,
        precision,
        numsteps,
        pipeline,
        log_dir,
        logstep):

    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
    else:
        devlist = devlist.split(',')

    datatype = "float%d" %precision


    # Generate the dataset and build the queue
    if pipeline=="feed_dict":
        timeUsed = benchmark_feed_dict.feed_dict_from_memory(
                batchsize,
                data_file,
                devlist,
                num_trainimg,
                imgsize,
                numsteps,
                log_dir,
                logstep,
                datatype)


    elif pipeline=="queue_runner":
        print('ToDo')


    elif pipeline=="dataset":
        if data_in_mem:
            timeUsed = benchmark_data_api.data_api_from_memory(
                    batchsize,
                    data_file,
                    devlist,
                    num_trainimg,
                    imgsize,
                    numsteps,
                    log_dir,
                    logstep,
                    datatype)
        else:
            timeUsed = benchmark_data_api.data_api_from_file(
                    batchsize,
                    data_file,
                    devlist,
                    num_trainimg,
                    imgsize,
                    numsteps,
                    log_dir,
                    logstep,
                    datatype)

    else:
        print("Pipeline has to be of feed_dict, queue_runner or dataset")
        return 0

    return timeUsed
