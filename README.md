# teapot

Here is my ML algorithm test lab. The best learning is by coding.

## 001

    $ python3 001_gd_mnist_fmnist.py

Comparing the training process and results for MNIST and FMNIST dataset by
pure GD and in a pure feed forward fully connected neural network, which means
to train by using the whole 60K train data for each epoch.

During the training, a plot would be displayed and updated in real time:

![image](https://github.com/xinlin-z/teapot/blob/master/pics/001_gd_mnist_fmnist.png)

All the hyper-parameters are the same for both networks.

Obviously, FMNIST is harder, but not for the first 60 epochs roughly.

