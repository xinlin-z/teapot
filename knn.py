import numpy as np
from tqdm import trange


def knn(x, y, t, nc, K):
    """Classification KNN.

    Use the same data format with dataset.load_mnist api.
    nc: number of classification
    """
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


def knn2(x, y, t, nc, K):
    colnum = len(t.T)
    colvs = []
    dists = []
    for i in trange(colnum):
        dist = np.linalg.norm(x-t[:,i:i+1], axis=0)
        cols = np.argpartition(dist,K)[:K]
        dists.append(dist[cols])
        colvs.append(np.argmax(y[:,cols],axis=0))
    nears = []
    for i in trange(colnum):
        near = [0 for i in range(nc)]
        for j in range(K):
            near[colvs[i][j]] += 1
        nears.append(near)
    rt = []
    for i in trange(colnum):
        tm = max(nears[i])
        if nears[i].count(tm) == 1:
            rt.append(nears[i].index(tm))
        else:
            mdist = []
            for j in range(nears[i].count(tm)):
                itm = nears[i].index(tm)
                tdist = 0
                for d,k in zip(dists[i],colvs[i]):
                    if k == itm:
                        tdist += d
                mdist.append((tdist,itm))
                nears[i][itm] = 0
            mdist.sort(key=lambda x:x[0])
            rt.append(mdist[0][1])
    return rt


