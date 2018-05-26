import numpy as np

from toolbox.sqDistance import *
from toolbox.oneOfK import *


class KnnModel():
    def fit(self, X, y, K, C=None):
        self.X = X
        self.y = y
        self.K = K
        if C is not None:
            self.C = C
        else:
            self.C = np.size(np.unique(y))

    def predict(self, Xtest):
        yhat, yprob = knnClassify(self.X, self.y, Xtest, self.K, self.C)
        return yhat, yprob


def knnClassify(Xtrain, ytrain, Xtest, K, C):
    Ntrain = Xtrain.shape[0]
    Nclasses = C
    if K > Ntrain:
        print("reducing K = %d to Ntrain = %d", K, Ntrain-1)
        K = Ntrain - 1
    dst = sqDistance(Xtest, Xtrain)
    ypred = np.zeros(Xtest.shape[0])
    if K == 1:
        closest = np.argmin(dst, axis=1)
        ypred = ytrain[closest]
        ypredProb, _ = oneOfK(ypred, Nclasses)
    else:
        closest = np.argsort(dst, axis=1)
        ypredProb = np.zeros((Xtest.shape[0], Nclasses))
        for i in range(Xtest.shape[0]):
            labels = ytrain[closest[i, 0:K]]
            hist, bin_edges = np.histogram(labels, bins=np.arange(1, Nclasses+2), density=True)
            ypredProb[i, :] = hist
        max = np.argmax(ypredProb, axis=1)
        ypred = max + 1
        ypred = ypred[:, np.newaxis]
    return ypred, ypredProb


if __name__ == '__main__':
    Xtrain = np.array([[1, 2], [11, 12], [21, 22], [3, 4], [13, 14], [23, 24]])
    Xtest = np.array([[2, 3], [12, 13], [22, 23]])
    ytrain = np.array([1, 2, 3, 1, 2, 3])
    ypred, ypredProb = knnClassify(Xtrain, ytrain, Xtest, 1, C=3)
    print("Done")