import numpy as np


def knn(x, y, t, K):
    """Classification KNN.

    Use the same data format with dataset.load_mnist api.
    """
    dist = np.linalg.norm(x-t, axis=0)
    col = np.argsort(dist)
    kset = [(dist[col[i]],np.argmax(y[:,col[i]])) for i in range(K)]
    near = [0 for i in range(len(y.T))]
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

