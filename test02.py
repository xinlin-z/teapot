import time
import numpy as np
import nnet
import dataset as ds


trd, ted = ds.get_mnist()
print('loaded 1...')
nn1 = nnet.fffnn((784,20,10))
t1 = time.time()
nn1.gd(trd, 0.1)
t2 = time.time()
print(round(t2-t1,2))


trdx2, trdy2, tedx2, tedy2 = ds.get_mnist2()
print('loaded 2...')
nn2 = nnet.fffnn((784,20,10))
t1 = time.time()
nn2.gd2(trdx2, trdy2, 0.1)
t2 = time.time()
print(round(t2-t1,2))


