# teapot

Here is my ML algorithm, numpy and matplotlib play ground.

The best way to learning is by coding. I will try to re-code the ML algorithms
which I've learned by python and numpy, and use matplotlib to plot in
decent ways.

## 0000

    $ python3 0000_show_mnist_fmnist.py

Show images in both MNIST and FMNIST dataset randomly. Each dataset has one
matplotlib figure window. The random choosing process covers both training set
and test set. So, we can be confident that the data is good!

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0000_show_mnist_fmnist.png)

Try to click on the window, it will restart the whole process and then you
can see another set of data image, which is also randomly choosed.

## 0010

    $ python3 0010_gd_mnist_fmnist.py

Comparing the training process and results for MNIST and FMNIST dataset by
pure GD and in a pure feed forward fully connected neural network, which means
to train by using the whole 60K train data for each epoch.

During the training, a plot would be displayed and updated in real time:

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0010_gd_mnist_fmnist.png)

All the hyper-parameters are the same for both networks.

Obviously, FMNIST is harder, but not for the first 60 epochs roughly. Why?

## 0020

    $ python3 0020_mean_distance_mnist_fmnist.py

Maybe the most simple and straight way to predict is to compute the L2 distance
between test image and mean training image. The one to whom gets the smallest
L2 distance is the prediction result.

![image](https://github.com/xinlin-z/teapot/blob/master/pics/0020_mean_distance_mnist_fmnist.png)

Again, FMNIST is harder. But this time, we could see the mean training images.

