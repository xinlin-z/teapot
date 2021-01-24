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
ctrdx, ctrdy, ctedx, ctedy = ds.load_cifar10()

# hyper-parameters
eta = 0.2
epoch = 1000
nn1 = (28*28,240,60,10)
nn2 = (28*28,240,60,10)
nn3 = (32*32*3,240,60,10)

# two same network for mnist and fmnsit
mnn = nnet.mlp(nn1, neuron='sigmoid', output='sigmoid', costfunc='cross_entropy')
fnn = nnet.mlp(nn2, neuron='sigmoid', output='sigmoid', costfunc='cross_entropy')
cnn = nnet.mlp(nn3, neuron='sigmoid', output='sigmoid', costfunc='cross_entropy')

# plot
plt.ion()
fig = plt.figure('Teapot : %s' % sys.argv[0], figsize=(10,6))
title = 'MNIST vs. FMNIST vs. CIFAR10 in full batch\n'
title += 'Eta:%.2f, Epoch:%d\n' % (eta,epoch)
title += 'MLP(MNIST): %s\n' % str(nn1)
title += 'MLP(FMNIST): %s\n' % str(nn2)
title += 'MLP(CIFAR10): %s' % str(nn3)
fig.suptitle(title)
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# rocking
epochs = []
amlist = []
aflist = []
aclist = []
mcost = []
fcost = []
ccost = []

ax1.plot(epochs, amlist, linewidth=0.5, color='r', label='MNIST')
ax1.plot(epochs, aflist, linewidth=0.5, color='b', label='FMNIST')
ax1.plot(epochs, aclist, linewidth=0.5, color='y', label='CIFAR10')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
line11, = ax1.plot(1, 1, linewidth=0.1, color='r')
line12, = ax1.plot(1, 1, linewidth=0.1, color='b')
line13, = ax1.plot(1, 1, linewidth=0.1, color='y')
line14, = ax1.plot(1, 1, linewidth=0.1, color='g')

ax2.plot(epochs, mcost, linewidth=0.5, color='r', label='MNIST')
ax2.plot(epochs, fcost, linewidth=0.5, color='b', label='FMNIST')
ax2.plot(epochs, ccost, linewidth=0.5, color='y', label='CIFAR10')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('cost')
line21, = ax1.plot(1, 1, linewidth=0.1, color='r')
line22, = ax1.plot(1, 1, linewidth=0.1, color='b')
line23, = ax1.plot(1, 1, linewidth=0.1, color='y')
line24, = ax1.plot(1, 1, linewidth=0.1, color='g')

for i in range(1,epoch+1):
    epochs.append(i)
    print('Epoch:', i)
    # mnist
    mnn.gd(mtrdx, mtrdy, eta)
    ma = mnn.ff(mtedx)
    test_result = np.argmax(ma, axis=0)
    right_result = np.argmax(mtedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    amlist.append(accuracy)
    cost = mnn.cost(mtedy, mtedx)
    mcost.append(cost)
    print('  mnist update, accuracy: %d, cost: %f' % (accuracy, cost))
    # fmnist
    fnn.gd(ftrdx, ftrdy, eta)
    fa = fnn.ff(ftedx)
    test_result = np.argmax(fa, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aflist.append(accuracy)
    cost = fnn.cost(ftedy, ftedx)
    fcost.append(cost)
    print(' fmnist update, accuracy: %d, cost: %f' % (accuracy, cost))
    # cifar10 
    cnn.gd(ctrdx, ctrdy, eta)
    ca = cnn.ff(ctedx)
    test_result = np.argmax(ca, axis=0)
    right_result = np.argmax(ctedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aclist.append(accuracy)
    cost = cnn.cost(ctedy, ctedx)
    ccost.append(cost)
    print('cifar10 update, accuracy: %d, cost: %f' % (accuracy, cost))
    # plot
    try:
        # mnist accuracy and tail line
        ax1.plot(epochs, amlist, linewidth=0.2, color='r')
        line11.remove()
        amlast = [amlist[-1] for i in epochs]
        line11, = ax1.plot(epochs, amlast, linewidth=0.1, color='r')
        # fmnist accuracy and tail line
        ax1.plot(epochs, aflist, linewidth=0.2, color='b')
        line12.remove()
        aflast = [aflist[-1] for i in epochs]
        line12, = ax1.plot(epochs, aflast, linewidth=0.1, color='b')
        # cifar10 accuracy and tail line
        ax1.plot(epochs, aclist, linewidth=0.2, color='y')
        line13.remove()
        aclast = [aclist[-1] for i in epochs]
        line13, = ax1.plot(epochs, aclast, linewidth=0.1, color='y')
        # ax1 vertical line
        line14.remove()
        line14, = ax1.plot(
            (epochs[-1], epochs[-1]),
            (0, max(amlist[-1], aflist[-1], aclist[-1])),
            linewidth = 0.1,
            color = 'g')
        # mnist cost and tail line
        ax2.plot(epochs, mcost, linewidth=0.2, color='r')
        line21.remove()
        mclast = [mcost[-1] for i in epochs]
        line21, = ax2.plot(epochs, mclast, linewidth=0.1, color='r')
        # fmnist cost and tail line
        ax2.plot(epochs, fcost, linewidth=0.2, color='b')
        line22.remove()
        fclast = [fcost[-1] for i in epochs]
        line22, = ax2.plot(epochs, fclast, linewidth=0.1, color='b')
        # cifar10 cost and tail line
        ax2.plot(epochs, ccost, linewidth=0.2, color='y')
        line23.remove()
        cclast = [ccost[-1] for i in epochs]
        line23, = ax2.plot(epochs, cclast, linewidth=0.1, color='y')
        # ax2 vertical line
        line24.remove()
        line24, = ax2.plot(
            (epochs[-1], epochs[-1]),
            (0, max(mcost[-1], fcost[-1], ccost[-1])),
            linewidth = 0.1,
            color = 'g')

        plt.pause(0.001)
    except Exception as e:
        print(repr(e))
        raise


print('Done!')
plt.ioff()
plt.show()

