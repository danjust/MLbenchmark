"""Functions to build a synthetic dataset of images
or import a .tfrecords file
"""

import tensorflow as tf
import numpy as np
import os
import sys

def generate_data(num_trainimg,num_testimg,imgsize,dtype):
    """Builds a synthetic dataset with horizontal or vertical lines.
    Returns images and labels.
    """
    trainimg = np.zeros([num_trainimg,imgsize,imgsize,1]).astype(eval('np.%s' %(dtype)))
    trainlabel = np.zeros([num_trainimg,2]).astype(np.int16)
    for i in range(num_trainimg):
        isvert = np.random.randint(2)
        line = np.random.randint(imgsize)
        if isvert==1:
            trainimg[i,:,line,0] = 1
            trainlabel[i,0] = 1
        else:
            trainimg[i,line,:,0] = 1
            trainlabel[i,1] = 1

    testimg = np.zeros([num_testimg,imgsize,imgsize,1]).astype(eval('np.%s' %(dtype)))
    testlabel = np.zeros([num_testimg,2]).astype(np.int16)
    for i in range(num_testimg):
        isvert = np.random.randint(2)
        line = np.random.randint(imgsize)
        if isvert==1:
            testimg[i,:,line,0] = 1
            testlabel[i,0] = 1
        else:
            testimg[i,line,:,0] = 1
            testlabel[i,1] = 1

    return trainimg, trainlabel, testimg, testlabel


def load_full_dataset(filename, width, height):
    """Imports full dataset from a .tfrecords file.
    Returns images and labels.
    """

    images = []
    labels =[]
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features
                             .feature['image']
                             .bytes_list
                             .value[0])

        img_1d = np.fromstring(img_string, dtype=np.float32)
        img = img_1d.reshape((height, width, -1))

        label = (example.features
                        .feature['label']
                        .int64_list
                        .value[0])

        images.append((img))
        labels.append((label))

    images = np.array(images)

    return images, labels


def _parse_function(example_proto):
    """Parse function for mapping data to a tf.data dataset object
    Return image and label.
    """
    features = {'image': tf.FixedLenFeature((), tf.string, default_value=""),
              'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    # Convert the image data from string back to the numbers
    # Note that the image is stored in a 1d array and needs to be reshaped later
    image = tf.decode_raw(parsed_features['image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(parsed_features['label'], tf.int32)
    return image, label


def get_iterator(filenames, batchsize, width, height, num_channels):
    dataset = tf.data.TFRecordDataset(filenames)
    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    # Repeat the input indefinitely.
    dataset = dataset.repeat()
    # Prepare batches
    dataset = dataset.batch(batchsize)
    # Create an iterator
    iterator = dataset.make_initializable_iterator()
    return iterator
