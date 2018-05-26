import numpy as np


def sqDistance(p, q, pSOS=None, qSOS=None):
    i = 0
    if pSOS is None:
        pSOS = np.sum(np.power(p, 2), axis=1)[:, np.newaxis]
    elif len(pSOS.shape) == 1:
        pSOS = pSOS[:, np.newaxis]
    if qSOS is None:
        qSOS = np.sum(np.power(q, 2), axis=1)[:, np.newaxis]
    elif len(qSOS.shape) == 1:
        qSOS = qSOS[:, np.newaxis]
    tmp1 = np.repeat(pSOS, qSOS.shape[0], axis=1)
    tmp2 = np.repeat(qSOS.T, pSOS.shape[0], axis=0)
    tmp3 = np.add(tmp1, tmp2)
    tmp4 = np.dot(p, q.T)
    d = tmp3 - 2*tmp4
    return d


if __name__ == '__main__':
    p = np.array([[1, 2, 3], [4, 5, 6]])
    q = np.array([[2, 2, 3], [4, 4, 5]])
    d = sqDistance(p, q)
    print(d)