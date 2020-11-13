import sys
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
from knn import knn2


K = 1
print('K =',K)


mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
print('load mnist ok...')
rt = knn2(mtrdx, mtrdy, mtedx, len(mtedx), K)
s = f = 0 
for i in range(len(mtedy.T)):
    t = np.argmax(mtedy[:,i])
    if rt[i] == t:
        s += 1
    else:
        f += 1
print('success:',s,'fail:',f)


ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()
print('load fmnist ok...')
rt = knn2(ftrdx, ftrdy, ftedx, len(ftedx), K)
s = f = 0 
for i in range(len(ftedy.T)):
    t = np.argmax(ftedy[:,i])
    if rt[i] == t:
        s += 1
    else:
        f += 1
print('success:',s,'fail:',f)



