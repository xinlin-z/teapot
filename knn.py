"""
KNN for classification


knn is single mode, input one data, return a single classification number.

knn2 is batch mode, input all test data, return a list.

knn3 is seperate mode, init with kmax which indicate the max accetable K,
and then compute the result list with one or more specific K. 

When there are more than one classification with the same max occurrences,
all three implementations of KNN would always choose the shortest one.

The input data format should be the same wth the return of load_mnist.
"""
import numpy as np
import copy
from tqdm import trange


def knn(x, y, t, K):
    """Classification KNN.

    Use the same data format with dataset.load_mnist api.
    """
    nc = len(y)
    dist = np.linalg.norm(x-t, axis=0)
    col = np.argpartition(dist, K)
    kset = [(dist[col[i]],np.argmax(y[:,col[i]])) for i in range(K)]
    near = [0 for i in range(nc)]
    for it in kset:
        near[it[1]] += 1
    tm = max(near)  # the max
    if near.count(tm) == 1:
        return near.index(tm)
    else:
        # choose the min dist one
        mdist = []  # dist of those maxes
        for i in range(near.count(tm)):
            itm = near.index(tm)
            tdist = 0
            for it in kset:
                if it[1] == itm:
                    tdist += it[0]
            mdist.append((tdist,itm))
            near[itm] = 0  # change near list
        mdist.sort(key=lambda x:x[0])
        return mdist[0][1]


def knn2(x, y, t, K):
    num = len(t.T)
    nc = len(y)
    colvs = []
    dists = []
    for i in trange(num):
        dist = np.linalg.norm(x-t[:,i:i+1], axis=0)
        cols = np.argpartition(dist,K)[:K]
        dists.append(dist[cols])
        colvs.append(np.argmax(y[:,cols],axis=0))
    nears = []
    for i in range(num):
        near = [0 for i in range(nc)]
        for j in range(K):
            near[colvs[i][j]] += 1
        nears.append(near)
    rt = []
    for i in range(num):
        tm = max(nears[i])
        if nears[i].count(tm) == 1:
            rt.append(nears[i].index(tm))
        else:
            mdist = []
            for j in range(nears[i].count(tm)):
                itm = nears[i].index(tm)
                tdist = 0
                for d,c in zip(dists[i],colvs[i]):
                    if c == itm:
                        tdist += d
                mdist.append((tdist,itm))
                nears[i][itm] = 0
            mdist.sort(key=lambda x:x[0])
            rt.append(mdist[0][1])
    return rt


class knn3():

    def __init__(self, x, y, t, kmax):
        if kmax > len(y.T):
            raise ValueError('kmax is bigger than training data.')
        self.num = len(t.T)
        self.nc = len(y)
        self.kmax = kmax
        self.nears = []
        for i in range(self.num):
            near = [0 for j in range(self.nc)]
            self.nears.append(near)
        self.colvs = []
        self.dists = []
        for i in trange(self.num):
            dist = np.linalg.norm(x-t[:,i:i+1], axis=0)
            cols = np.argsort(dist)[:kmax]  # not np.argpartition anymore
            self.dists.append(dist[cols])
            self.colvs.append(np.argmax(y[:,cols],axis=0))

    def compute(self, K):
        if K > self.kmax or K < 1:
            raise ValueError('K should le kmax and gt zero.')
        nears = copy.deepcopy(self.nears)
        for i in range(self.num):
            for j in range(K):
                nears[i][self.colvs[i][j]] += 1
        rt = []
        for i in range(self.num):
            tm = max(nears[i])
            if nears[i].count(tm) == 1:
                rt.append(nears[i].index(tm))
            else:
                mdist = []
                for j in range(nears[i].count(tm)):
                    itm = nears[i].index(tm)
                    tdist = 0
                    for d,c in zip(self.dists[i][:K],self.colvs[i][:K]):
                        if c == itm:
                            tdist += d
                    mdist.append((tdist,itm))
                    nears[i][itm] = 0
                mdist.sort(key=lambda x:x[0])
                rt.append(mdist[0][1])
        return rt


