import sys
import numpy as np
import nnet
import dataset as ds
import time
import func
import matplotlib.pyplot as plt


# set float type and load data 
ds.DTYPE = np.float64
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()
ttrdx, ttrdy, ttedx, ttedy = ds.tload_fmnist()

# hyper-parameters
eta = 0.2
epoch = 1000
nn1 = (28*28,240,60,10)
nn2 = (28*28,240,60,10)
nn3 = (28*28,240,60,10)
nn4 = (28*28,240,60,10)

# networks
mnn1 = nnet.mlp(nn1, neuron='sigmoid', output='sigmoid', costfunc='quadratic')
mnn2 = nnet.mlp(nn2, neuron='sigmoid', output='sigmoid', costfunc='cross_entropy')
mnn3 = nnet.mlp(nn3, neuron='sigmoid', output='softmax', costfunc='log_likelihood')
mnn4 = nnet.mlp(nn4, neuron='tanh', output='softmax', costfunc='log_likelihood')

# plot
plt.ion()
fig = plt.figure('Teapot : %s' % sys.argv[0], figsize=(10,6))
title = 'FMNIST with diff config in full batch\n'
title += 'Eta:%.2f, Epoch:%d\n' % (eta,epoch)
title += 'MLP(1): %s + sigmoid + quadratic \n' % str(nn1)
title += 'MLP(2): %s + sigmoid + cross_entropy \n' % str(nn2)
title += 'MLP(3): %s + sigmoid + softmax + log_likelihood \n' % str(nn3)
title += 'MLP(4): %s + tanh + softmax + log_likelihood \n' % str(nn4)
fig.suptitle(title)
fig.subplots_adjust(top=0.75)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# rocking
epochs = []
amlist = []
aflist = []
aclist = []
atlist = []
mcost = []
fcost = []
ccost = []
tcost = []

ax1.plot(epochs, amlist, linewidth=0.5, color='r', label='sigmoid+quadratic')
ax1.plot(epochs, aflist, linewidth=0.5, color='b', label='sigmoid+cross_entropy')
ax1.plot(epochs, aclist, linewidth=0.5, color='y', label='sigmoid+softmax+log_likelihood')
ax1.plot(epochs, atlist, linewidth=0.5, color='c', label='tanh+softmax+log_likelihood')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
line11, = ax1.plot(1, 1, linewidth=0.1, color='r')
line12, = ax1.plot(1, 1, linewidth=0.1, color='b')
line13, = ax1.plot(1, 1, linewidth=0.1, color='y')
line14, = ax1.plot(1, 1, linewidth=0.1, color='c')
line15, = ax1.plot(1, 1, linewidth=0.1, color='g')

ax2.plot(epochs, mcost, linewidth=0.5, color='r', label='sigmoid+quadratic')
ax2.plot(epochs, fcost, linewidth=0.5, color='b', label='sigmoid+cross_entropy')
ax2.plot(epochs, ccost, linewidth=0.5, color='y', label='sigmoid+softmax+log_likelihood')
ax2.plot(epochs, tcost, linewidth=0.5, color='c', label='tanh+softmax+log_likelihood')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('test cost')
line21, = ax1.plot(1, 1, linewidth=0.1, color='r')
line22, = ax1.plot(1, 1, linewidth=0.1, color='b')
line23, = ax1.plot(1, 1, linewidth=0.1, color='y')
line24, = ax1.plot(1, 1, linewidth=0.1, color='c')
line25, = ax1.plot(1, 1, linewidth=0.1, color='g')

for i in range(1,epoch+1):
    epochs.append(i)
    print('Epoch:', i)
    # mnn1
    mnn1.gd(ftrdx, ftrdy, eta)
    ma = mnn1.ff(ftedx)
    test_result = np.argmax(ma, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    amlist.append(accuracy)
    cost = mnn1.cost(ftedy, ma)
    mcost.append(cost)
    print(' mnn1 update, accuracy: %d, cost: %f' % (accuracy, cost))
    # mnn2
    mnn2.gd(ftrdx, ftrdy, eta)
    fa = mnn2.ff(ftedx)
    test_result = np.argmax(fa, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aflist.append(accuracy)
    cost = mnn2.cost(ftedy, fa)
    fcost.append(cost)
    print(' mnn2 update, accuracy: %d, cost: %f' % (accuracy, cost))
    # mnn3 
    mnn3.gd(ftrdx, ftrdy, eta)
    ca = mnn3.ff(ftedx)
    test_result = np.argmax(ca, axis=0)
    right_result = np.argmax(ftedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    aclist.append(accuracy)
    cost = mnn3.cost(ftedy, ca)
    ccost.append(cost)
    print(' mnn3 update, accuracy: %d, cost: %f' % (accuracy, cost))
    # mnn4 
    mnn4.gd(ttrdx, ttrdy, eta)
    ta = mnn4.ff(ttedx)
    test_result = np.argmax(ta, axis=0)
    right_result = np.argmax(ttedy, axis=0)
    accuracy = sum([int(x==y) for x,y in zip(test_result,right_result)])
    atlist.append(accuracy)
    cost = mnn4.cost(ttedy, ta)
    tcost.append(cost)
    print(' mnn4 update, accuracy: %d, cost: %f' % (accuracy, cost))
    # plot
    try:
        # mnn1 accuracy and tail line
        ax1.plot(epochs, amlist, linewidth=0.2, color='r')
        line11.remove()
        amlast = [amlist[-1] for i in epochs]
        line11, = ax1.plot(epochs, amlast, linewidth=0.1, color='r')
        # mnn2 accuracy and tail line
        ax1.plot(epochs, aflist, linewidth=0.2, color='b')
        line12.remove()
        aflast = [aflist[-1] for i in epochs]
        line12, = ax1.plot(epochs, aflast, linewidth=0.1, color='b')
        # mnn3 accuracy and tail line
        ax1.plot(epochs, aclist, linewidth=0.2, color='y')
        line13.remove()
        aclast = [aclist[-1] for i in epochs]
        line13, = ax1.plot(epochs, aclast, linewidth=0.1, color='y')
        # mnn4 accuracy and tail line
        ax1.plot(epochs, atlist, linewidth=0.2, color='c')
        line14.remove()
        atlast = [atlist[-1] for i in epochs]
        line14, = ax1.plot(epochs, atlast, linewidth=0.1, color='c')
        # ax1 vertical line
        line15.remove()
        line15, = ax1.plot(
            (epochs[-1], epochs[-1]),
            (0, max(amlist[-1], aflist[-1], aclist[-1], atlist[-1])),
            linewidth = 0.1,
            color = 'g')
        # mnn1 cost and tail line
        ax2.plot(epochs, mcost, linewidth=0.2, color='r')
        line21.remove()
        mclast = [mcost[-1] for i in epochs]
        line21, = ax2.plot(epochs, mclast, linewidth=0.1, color='r')
        # mnn2 cost and tail line
        ax2.plot(epochs, fcost, linewidth=0.2, color='b')
        line22.remove()
        fclast = [fcost[-1] for i in epochs]
        line22, = ax2.plot(epochs, fclast, linewidth=0.1, color='b')
        # mnn3 cost and tail line
        ax2.plot(epochs, ccost, linewidth=0.2, color='y')
        line23.remove()
        cclast = [ccost[-1] for i in epochs]
        line23, = ax2.plot(epochs, cclast, linewidth=0.1, color='y')
        # mnn4 cost and tail line
        ax2.plot(epochs, tcost, linewidth=0.2, color='c')
        line24.remove()
        tclast = [tcost[-1] for i in epochs]
        line24, = ax2.plot(epochs, tclast, linewidth=0.1, color='c')
        # ax2 vertical line
        line25.remove()
        line25, = ax2.plot(
            (epochs[-1], epochs[-1]),
            (0, max(mcost[-1], fcost[-1], ccost[-1], tcost[-1])),
            linewidth = 0.1,
            color = 'g')

        plt.pause(0.001)
    except Exception as e:
        print(repr(e))
        raise


print('Done!')
plt.ioff()
plt.show()

