import sys
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
from knn import knn3

# plot prepare
plt.ion()
fig = plt.figure('Teapot : KNN test')
fig.suptitle('KNN test on MNIST & FMNIST')
ax = fig.add_subplot()
ax.set_xlabel('K')
ax.set_ylabel('Scores')
ax.plot([], [], color='r', label='MNIST')
ax.plot([], [], color='b', label='FMNIST')
ax.legend()

# set kmax
kmax = 500
print('kmax =',kmax)
rr = 10000

# load mnist
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
print('load mnist ok...')

# init
print('init...')
knn = knn3(mtrdx, mtrdy, mtedx[:,0:rr], kmax)

# compute
mk = []
succ_m = []
for k in range(1,kmax+1):
    print('knn mnist compute K = %d:' % k)
    rt = knn.compute(k)
    s = 0
    for i in range(rr):
        t = np.argmax(mtedy[:,i])
        if rt[i] == t:
            s += 1
    print('success:',s)
    succ_m.append(s)
    mk.append(k)
    ax.plot(mk, succ_m, linewidth=0.2, color='r')
    plt.pause(0.001)

maxk_m = succ_m.index(max(succ_m)) + 1
ax.plot((maxk_m,maxk_m), (int(succ_m[maxk_m]*0.95),int(succ_m[maxk_m]*1.05)),
                                linewidth=0.2, color='r')
plt.pause(0.001)

# load fmnist
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()
print('load fmnist ok...')

# init
print('init...')
knn = knn3(ftrdx, ftrdy, ftedx[:,0:rr], kmax)

# compute
fk = []
succ_f = []
for k in range(1,kmax+1):
    print('knn fmnist compute K = %d:' % k)
    rt = knn.compute(k)
    s = 0
    for i in range(rr):
        t = np.argmax(ftedy[:,i])
        if rt[i] == t:
            s += 1
    print('success:', s)
    succ_f.append(s)
    fk.append(k)
    ax.plot(fk, succ_f, linewidth=0.2, color='b')
    plt.pause(0.001)

maxk_f = succ_f.index(max(succ_f)) + 1
ax.plot((maxk_f,maxk_f), (int(succ_f[maxk_f]*0.95),int(succ_f[maxk_f]*1.05)),
                                linewidth=0.2, color='b')
plt.pause(0.001)

print('Done!')
plt.ioff()
plt.show()
