import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(img,label,filename):
    num_img = img.shape[0]
    with tf.python_io.TFRecordWriter(filename) as writer:
        for ind in range(num_img):
            feature = {
                    'label': _int64_feature(label[ind]),
                    'image': _bytes_feature(img[ind].tostring())
            }
            example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())


def unpickle3(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle2(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def unpickle(file):
    if sys.version_info.major==3:
        dict = unpickle3(file)
    elif sys.version_info.major==2:
        dict = unpickle2(file)
    return dict

def import_cifar(data_dir):
    # Training dataset
    trainimg = np.empty([50000,3072])
    lbl = np.empty(50000)
    trainlabel = np.zeros([50000,10])

    for i in range(1,6):
        filename = os.path.join(data_dir, 'data_batch_%d' % i)
        d = unpickle(filename)
        trainimg[(i-1)*10000:i*10000,:] = d[b'data']
        lbl[(i-1)*10000:i*10000] = d[b'labels']

    trainimg = trainimg.reshape([50000,3,32,32])
    trainimg = np.transpose(trainimg, (0, 2, 3, 1))
    trainlabel[np.arange(50000), np.int8(lbl)] = 1

    # Validation dataset
    testlabel = np.zeros([10000,10])
    filename = os.path.join(data_dir, 'test_batch')
    d = unpickle(filename)
    testimg = d[b'data']
    lbl = d[b'labels']

    testimg = testimg.reshape([10000,3,32,32])
    testimg = np.transpose(testimg, (0, 2, 3, 1))
    testlabel[np.arange(10000), np.int8(lbl)] = 1

    return trainimg, trainlabel, testimg, testlabel
