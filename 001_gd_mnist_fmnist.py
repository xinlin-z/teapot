import sys
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
mnn.load(
    activation_func =    func.sigmoid,
    activation_func_dz = func.sigmoid_dz,
    cost_func =          func.cross_entropy_x,
    cost_func_da =       func.cross_entropy_x_da
)

fnn = nnet.fffnn(nn_size)
fnn.load(
    activation_func =    func.sigmoid,
    activation_func_dz = func.sigmoid_dz,
    cost_func =          func.cross_entropy_x,
    cost_func_da =       func.cross_entropy_x_da
)

# plot
plt.ion()
fig = plt.figure('Teapot : %s' % sys.argv[0])
fig.suptitle('MNIST vs. FMNIST')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# rocking
epochs = []
amlist = []
aflist = []
mcost = []
fcost = []

ax1.set_title('NN:%s, Learning Rate:%d, Epoch:%d'%(str(nn_size),eta,epoch))
ax1.plot(epochs, amlist, linewidth=0.5, color='r', label='MNIST')
ax1.plot(epochs, aflist, linewidth=0.5, color='b', label='FMNIST')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')

ax2.plot(epochs, mcost, linewidth=0.5, color='r', label='MNIST')
ax2.plot(epochs, fcost, linewidth=0.5, color='b', label='FMNIST')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('cost')

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
    mcost.append(cost)
    print(' mnist update, accuracy: %d, cost: %f' % (accuracy, cost))
    # fmnist
    fnn.gd(ftrdx, ftrdy, eta)
    fa = fnn.ff(ftedx)
    test_result = np.argmax(fa, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aflist.append(accuracy)
    cost = fnn.cost(ftedy, ftedx)
    fcost.append(cost)
    print('fmnist update, accuracy: %d, cost: %f' % (accuracy, cost))
    # plot
    try:
        ax1.plot(epochs, amlist, linewidth=0.5, color='r', label='MNIST')
        ax1.plot(epochs, aflist, linewidth=0.5, color='b', label='FMNIST')
        ax2.plot(epochs, mcost,  linewidth=0.5, color='r', label='MNIST')
        ax2.plot(epochs, fcost,  linewidth=0.5, color='b', label='FMNIST')
        plt.pause(0.001)
    except:
        ...


print('Done!')
plt.ioff()
plt.show()

