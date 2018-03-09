FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git

RUN mkdir /model
WORKDIR /model
ADD . /model

WORKDIR /model/MLbenchmark

CMD ["python", "benchmark.py", \
    "--testMatMul=True", \
    "--testConv=True", \
    "--testRNN=False", \
    "--testCNN=True", \
    "--num_gpu=$NUM_GPU", \
    "--matsize=8192", \
    "--kernelsize=15", \
    "--iter=10", \
    "--rnn_type=lstm", \
    "--iter_rnn=1000", \
    "--numsteps_cnn=10000", \
    "--logstep_cnn=250"]
