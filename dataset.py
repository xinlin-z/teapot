"""
train data and test data are separated as two data.
For each data, it is a list with tuples like (x,y).
For each tuple in list, x and y are both numpy ndarray.
x represents input, y represents output correspondingly.
The dtype of ndarray for x and y are both float64 (for numba).
The position(index) with 1.0(for both sigmoid and tanh) value in vector
y indicates the corect classification info.

Best practice is to normalize the input of the network between 0 and 1
if sigmoid is the activation function and -0.5 and 0.5 if tanh is the
activation function. (not for labeled output)
ReLU is the same with sigmoid.

If you want to have an extra validation data, please help youself
by making a slice from tr_d. This is called "hold out" method.

"""

import pickle
import gzip
import os
import random
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH_PREFIX = 'data/'
DTYPE = np.float64


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(DATA_PATH_PREFIX+'mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = \
                                pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    Listed all return data, so we can read them more convennience.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x,(784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x,(784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x,(784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (list(training_data), list(validation_data), list(test_data))


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def show_mnist(imgdata):
    """mnist imgdata should be ndarray which shape is (784,1)"""
    plt.imshow(imgdata.reshape(28,28), cmap='gray')
    plt.show()
    #plt.draw()
    #plt.pause(0.001)
    return


##############################
# mnist data.
##############################
def get_mnist():
    """ load mnist in customized float type, 0-1 """
    tr,va,te = load_data_wrapper()
    tr.extend(va)
    for i in range(len(tr)):
        tr[i][0].astype(DTYPE)
        tr[i][1].astype(DTYPE)
    for i in range(len(te)):
        te[i][0].astype(DTYPE)
        te[i][1].astype(DTYPE)
    return tr, te


def load_mnist():
    """trdx, trdy, tedx, tedy = load_mnist() """
    trd, vad, ted = load_data()
    trdi2 = np.hstack([np.reshape(x,(784,1)) for x in trd[0]])
    vadi2 = np.hstack([np.reshape(x,(784,1)) for x in vad[0]])
    trdi2 = np.hstack((trdi2,vadi2))
    trdl2 = np.hstack([vectorized_result(y) for y in trd[1]])
    vadl2 = np.hstack([vectorized_result(y) for y in vad[1]])
    trdl2 = np.hstack((trdl2,vadl2))
    tedi2 = np.hstack([np.reshape(x,(784,1)) for x in ted[0]])
    tedl2 = np.hstack([vectorized_result(y) for y in ted[1]])
    return (trdi2.astype(DTYPE),
            trdl2.astype(DTYPE),
            tedi2.astype(DTYPE),
            tedl2.astype(DTYPE))


def get_mnist_tanh():
    tr, te = get_mnist()
    trd = []
    # ted = []
    for i in range(len(tr)):
        a = tr[i][0] - 0.5
        # b = tr[i][1] - 0.5
        trd.append((a,tr[i][1]))
    return trd, te

def get_mnist_cnn():
    tr, te = get_mnist()
    trc = []
    for i in range(len(tr)):
        x = tr[i][0]
        y = tr[i][1]
        trc.append((x.reshape(1,28,28),y))
    tec = []
    for j in range(len(te)):
        x = te[j][0]
        y = te[j][1]
        tec.append((x.reshape(1,28,28),y))
    return trc, tec


##############################
# fasion mnist data.
##############################
def _load_fmnist(kind):

    labels_path = os.path.join(DATA_PATH_PREFIX+'fmnist',
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(DATA_PATH_PREFIX+'fmnist',
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_fmnist():
    tr_dx, tr_dy = _load_fmnist('train')
    te_dx, te_dy = _load_fmnist('t10k')
    tr_d = [(np.reshape(x.astype(DTYPE),(784,1))/255,
                vectorized_result(y)) for x,y in zip(tr_dx, tr_dy)]
    te_d = [(np.reshape(x.astype(DTYPE),(784,1))/255,
                vectorized_result(y)) for x,y in zip(te_dx, te_dy)]
    return tr_d, te_d


def load_fmnist():
    trd, ted = get_fmnist()
    trdx = np.hstack([x[0] for x in trd])
    trdy = np.hstack([x[1] for x in trd])
    tedx = np.hstack([x[0] for x in ted])
    tedy = np.hstack([x[1] for x in ted])
    return trdx, trdy, tedx, tedy


def get_fmnist_tanh():
    tr, te = get_fmnist()
    trd = []
    for i in range(len(tr)):
        a = tr[i][0] - 0.5
        trd.append((a,tr[i][1]))
    return trd, te


#######################
# CIFAR10
#######################
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
    return dict


def get_cifar10():

    tr_d = []
    dd = unpickle(DATA_PATH_PREFIX+'cifar10/data_batch_1')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255  # channels * tall * width
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        tr_d.append((a,b))

    dd = unpickle(DATA_PATH_PREFIX+'cifar10/data_batch_2')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        tr_d.append((a,b))

    dd = unpickle(DATA_PATH_PREFIX+'cifar10/data_batch_3')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        tr_d.append((a,b))

    dd = unpickle(DATA_PATH_PREFIX+'cifar10/data_batch_4')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        tr_d.append((a,b))

    dd = unpickle(DATA_PATH_PREFIX+'cifar10/data_batch_5')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        tr_d.append((a,b))

    te_d = []
    dd = unpickle(DATA_PATH_PREFIX+'cifar10/test_batch')
    for i in range(len(dd['data'])):
        a = dd['data'][i].reshape(3,32,32)/255
        a = a.astype(DATA_FLOAT_TYPE)
        b = vectorized_result(dd['labels'][i])
        te_d.append((a,b))

    return tr_d, te_d



