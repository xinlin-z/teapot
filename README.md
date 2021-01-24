# Contents

* [INTRO teapot](#INTRO-teapot)
* [SHOW](#SHOW)
    * [0000 show MNIST & FMNIST & CIFAR10 data images randomly](#0000-show-MNIST--FMNIST--CIFAR10-data-images-randomly)
    * [0001 show neuron activation & derivative](#0001-show-neuron-activation--derivative)
    * [0002 show IRIS data in 3D](#0002-show-IRIS-data-in-3D)
* [MLP](#MLP)
    * [0010 train MNIST & FMNIST & CIFAR10 in full batch](#0010-train-MNIST--FMNIST--CIFAR10-in-full-batch)
* [MEAN](#MEAN)
    * [0020 compare distance to mean image on MNIST and FMNIST](#0020-compare-distance-to-mean-image-on-MNIST-and-FMNIST)
* [KNN](#KNN)
    * [0030 compare KNN on MNIST and FMNIST](#0030-compare-KNN-on-MNIST-and-FMNIST)
    * [0032 find the best K value on MNIST and FMNIST](#0032-find-the-best-K-value-on-MNIST-and-FMNIST)

# INTRO teapot

Here is my ML algorithm, numpy and matplotlib play ground.

The best way to learning is by coding. I will try to re-code the ML algorithms
which I've learned by python and numpy, and use matplotlib to plot as much as
possible in decent ways. Yes, I am trying to reinvent the wheels.

# SHOW

## 0000 show MNIST & FMNIST & CIFAR10 data images randomly

    $ python3 0000_show_mnist_fmnist.py

Show images in both MNIST, FMNIST and CIFAR10 dataset randomly. Each dataset
has one matplotlib figure window. The random choosing process covers both
training set and test set. So, we can be confident that the data is good!

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0000_show_mnist_fmnist_cifar10.png)

Click on the window, it will restart the whole process and then you can see
another set of data image, which is also randomly choosed.

## 0001 show neuron activation & derivative

    $ python3 0001_show_neuron.py

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0001_show_neuron.png)

## 0002 show IRIS data in 3D

    $ python3 0002_show_iris.py

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0002_show_iris.png)

It's obvious that different flowers belong to different spaces.

# MLP

## 0010 train MNIST & FMNIST & CIFAR10 in full batch

    $ python3 0010_mlp_mnist_fmnist_cifar10.py

Comparing the training process and results for MNIST, FMNIST and CIFAR10
dataset by MLP neural network in full batch, which means to train by using
the whole 60K train data (50K for CIFAR10) for each epoch.

During the training, a plot would be displayed and updated in real time:

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0010_mlp_mnist_fmnist_cifar10.png)

All the hyper-parameters are the same.

## 0011 train FMNIST with different config in MLP network

    $ python3 0011_mlp_fmnist_diff_config.py

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0011_mlp_fmnist_diff_config.png)

# MEAN

## 0020 compare distance to mean image on MNIST and FMNIST

    $ python3 0020_mean_distance_mnist_fmnist.py

Maybe the most simple and straight way to predict is to compute the L2 distance
between test image and mean training image. The one to whom gets the smallest
L2 distance is the prediction result.

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0020_mean_distance_mnist_fmnist.png)

Again, FMNIST is harder. But this time, we could see the mean training images.

# KNN

## 0030 compare KNN on MNIST and FMNIST

     $ python3 0030_knn_mnist_fmnist.py

There is no learning process in KNN algorithm, so it's not a learning method
and more computation load is need while prediction. Below is the whole result
of using KNN on MNIST and FMNIST datasets:

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0030_knn_mnist_fmnist.png)

This plot would be updated in real time while running the above python cmd.

## 0032 find the best K value on MNIST and FMNIST

    $ python3 0032_knn3_test_diff_K.py

I just want to know the best K value for MNIST and FMNIST dataset, so here
it is:

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0032_knn3_test_diff_K.png)

When K=3, MNIST gets 9717 right classifications, which is the best. When K=4,
FMNIST gets 8596 right classifications, which is the best.

K value should be surprisingly small...


