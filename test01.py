import numpy as np
import nnet
import dataset as ds
import time
import func


# set float type and load data
ds.DTYPE = np.float64
trd, ted = ds.get_mnist()
print('mnist dataset is loaded, %s, 0-1, train=%d, test=%d' 
        % (ds.DTYPE, len(trd),len(ted)))

# hyper-parameters
sizes = (784,20,10)
nn = nnet.fffnn(sizes)
eta = 0.01  # learning rate
mblen = 10  # mini batch length
epoch = 100000

# record files
passed_file = 'passed.txt'
cost_file = 'cost.txt'
with open(passed_file,'a') as pf, \
        open(cost_file,'a') as cf:
    write_time = func.getDateTime()
    pf.write('---- new training start at %s ----\n'%write_time)
    pf.write('nnet sizes: %s\n'%str(sizes))
    cf.write('---- new training start at %s ----\n'%write_time)
    cf.write('nnet sizes: %s\n'%str(sizes))
    cf.write('cost(test),cost(train)\n')

for i in range(epoch):
    t1 = time.time()
    nn.sgd(trd, mblen, eta)
    t2 = time.time()
    #
    test_result = [(np.argmax(nn.ff(x)),np.argmax(y)) for x,y in ted]
    passed = sum([int(x==y) for x,y in test_result])
    cost = nn.cost(trd)
    cost2 = nn.cost(ted)
    with open(passed_file,'a') as pf, \
            open(cost_file,'a') as cf:
        pf.write(str(passed)+'\n')
        cf.write(str(cost2)+','+str(cost)+'\n')
    t3 = time.time()
    print('Epoch {}: {}/{}, Cost(test):{}, Cost(train):{}, ' 
          'Round time(s):{}(+{})'.format(i+1,
                                    passed,
                                    len(ted),
                                    round(cost2,8),
                                    round(cost,8),
                                    round(t2-t1,2),
                                    round(t3-t2,2)))
