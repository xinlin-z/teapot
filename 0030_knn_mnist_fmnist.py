import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
from knn import knn


# dataset load
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()

K = 10
xlabels = ('MNIST','FMNIST')
x = np.arange(len(xlabels))
width = 0.3

fig = plt.figure('Teapot: KNN on MNIST & FMNIST')
fig.suptitle('KNN on MNIST & FMNIST')
ax = fig.add_subplot()
perf_s = [0,0]
perf_f = [0,0]
rects1 = ax.bar(x-width/2, perf_s, 0.4, color='g', alpha=0.7, label='Success')
rects2 = ax.bar(x+width/2, perf_f, 0.4, color='r', alpha=0.7, label='Fail')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.legend()
ax.set_title('K = %d' % K)
fig.tight_layout()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    anno = []
    for rect in rects:
        height = rect.get_height()
        a = ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 1 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        anno.append(a)
    return anno


anno1 = autolabel(rects1)
anno2 = autolabel(rects2)
plt.ion()

for i in range(10000):
    print('index of column in test set:[%d]' % i)
    p = knn(mtrdx, mtrdy, mtedx[:,i:i+1], K)
    t = np.argmax(mtedy[:,i])
    if p == t:
        perf_s[0] += 1
    else:
        perf_f[0] += 1
    print(' mnist, prediction=%d, truth=%d, %s'
                    %(p,t,'Success' if p==t else 'Fail'))

    p = knn(ftrdx, ftrdy, ftedx[:,i:i+1], K)
    t = np.argmax(ftedy[:,i])
    if p == t:
        perf_s[1] += 1
    else:
        perf_f[1] += 1
    print('fmnist, prediction=%d, truth=%d, %s'
                    %(p,t,'Success' if p==t else 'Fail'))

    [(x.remove(),y.remove()) for x,y in zip(rects1,rects2)]
    rects1 = ax.bar(x-width/2, perf_s, 0.4, color='g', alpha=0.7)
    rects2 = ax.bar(x+width/2, perf_f, 0.4, color='r', alpha=0.7)
    [(x.remove(),y.remove()) for x,y in zip(anno1,anno2)]
    anno1 = autolabel(rects1)
    anno2 = autolabel(rects2)

    plt.pause(0.001)


print('Done!')
plt.ioff()
plt.show()


