# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf

from utils_tf.cifar10_benchmark import cifar10, cifar10_input


def train(data_dir,train_dir,batch_size,max_steps):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      binary_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
      images, labels = cifar10_input.inputs(eval_data=False,
                                              data_dir=binary_dir,
                                              batch_size=batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images,batch_size)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step,batch_size)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=max_steps),
               tf.train.NanTensorHook(loss)],
        config=tf.ConfigProto(
            log_device_placement=False)) as mon_sess:

      print("========================================\n")
      print("start training cnn")
      tstart = time.time()
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
      tstop = time.time()
      dur = tstop-tstart
      print("========================================\n")
    return dur
