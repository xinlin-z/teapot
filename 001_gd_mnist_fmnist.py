import numpy as np
import nnet
import dataset as ds
import time
import func
import matplotlib.pyplot as plt


# set float type and load data 
ds.DTYPE = np.float64
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()

# hyper-parameters
eta = 1 
epoch = 1000
nn_size = (784,120,10)

# two same network for mnist and fmnsit
mnn = nnet.fffnn(nn_size)
mnn.load(activation_func =    func.sigmoid,
         activation_func_dz = func.sigmoid_dz,
         cost_func =          func.cross_entropy_x,
         cost_func_da =       func.cross_entropy_x_da)
fnn = nnet.fffnn(nn_size)
fnn.load(activation_func =    func.sigmoid,
         activation_func_dz = func.sigmoid_dz,
         cost_func =          func.cross_entropy_x,
         cost_func_da =       func.cross_entropy_x_da)

# plot
plt.ion()
fig = plt.figure('teapot 001')
fig.suptitle('MNIST vs. FMNIST')
ax = fig.add_subplot()

# rocking
epochs = []
amlist = []
aflist = []
ax.set_title('NN:%s, Learning Rate:%d, Epoch:%d'%(str(nn_size),eta,epoch))
ax.plot(epochs, amlist, linewidth=0.5, color='r', label='MNIST')
ax.plot(epochs, aflist, linewidth=0.5, color='b', label='FMNIST')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

for i in range(1,epoch):
    epochs.append(i)
    print('epoch:', i)
    # mnist
    mnn.gd(mtrdx, mtrdy, eta)
    ma = mnn.ff(mtedx)
    test_result = np.argmax(ma, axis=0)
    right_result = np.argmax(mtedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    amlist.append(accuracy)
    cost = mnn.cost(mtedy, mtedx)
    print(' mnist update, accuracy: %d, cost: %f' % (accuracy, cost))
    # fmnist
    fnn.gd(ftrdx, ftrdy, eta)
    fa = fnn.ff(ftedx)
    test_result = np.argmax(fa, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aflist.append(accuracy)
    cost = fnn.cost(ftedy, ftedx)
    print('fmnist update, accuracy: %d, cost: %f' % ( accuracy, cost))
    # plot
    try:
        ax.plot(epochs, amlist, linewidth=0.5, color='r', label='MNIST')
        ax.plot(epochs, aflist, linewidth=0.5, color='b', label='FMNIST')
        plt.pause(0.001)
    except:
        ...


print('end')
plt.ioff()
plt.show()

