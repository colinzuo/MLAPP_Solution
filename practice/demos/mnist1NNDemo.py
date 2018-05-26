import os
import scipy.io
import numpy as np
from datetime import datetime
from pyflann import *

from toolbox.sqDistance import *


def mnist1NNDemo(test_size=1000, use_flann=False, perform_shuffle=False):
    data_path = os.path.join("..", "bigData", "mnistAll", "mnistAll.mat")
    mat = scipy.io.loadmat(data_path)

    trainndx = range(0, 60000)
    testndx = range(0, test_size)

    ntrain = len(trainndx)
    ntest = len(testndx)

    Xtrain = np.reshape(mat['mnist']['train_images'][0][0][:, :, trainndx], (28*28, ntrain), order='A').T.astype(int)
    Xtest = np.reshape(mat['mnist']['test_images'][0][0][:, :, testndx], (28*28, ntest), order='A').T.astype(int)

    if perform_shuffle:
        perm = np.random.permutation(range(28*28))
        Xtrain = np.ascontiguousarray(Xtrain[:, perm], dtype=int)
        Xtest = np.ascontiguousarray(Xtest[:, perm], dtype=int)
        print("Data     Shuffled")

    ytrain = mat['mnist']['train_labels'][0][0][trainndx]
    ytest = mat['mnist']['test_labels'][0][0][testndx]
    ypred = np.zeros(ytest.shape, dtype=np.uint8)

    if use_flann:
        start_time1 = datetime.now()
        flann = FLANN()
        index_path = os.path.join("..", "tmp", "mnist.index")
        if os.path.exists(index_path):
            flann.load_index(index_path, Xtrain)
        else:
            params = flann.build_index(Xtrain, algorithm="autotuned", target_precision=0.99)
            flann.save_index(index_path)
        start_time = datetime.now()
    else:
        XtrainSOS = np.sum(np.power(Xtrain, 2), axis=1)
        start_time = datetime.now()
        XtestSOS = np.sum(np.power(Xtest, 2), axis=1)

    batch_size = 200
    nbatch = (ntest + batch_size - 1) // batch_size

    for i in range(nbatch):
        batch_begin = i * batch_size
        batch_end = (i + 1) * batch_size
        Xtest_batch = Xtest[batch_begin:batch_end, :]
        if use_flann:
            result, dists = flann.nn_index(Xtest_batch, 1)
            ypred[batch_begin:batch_end] = ytrain[result]
        else:
            XtestSOS_batch = XtestSOS[batch_begin:batch_end]
            dst = sqDistance(Xtest_batch, Xtrain, XtestSOS_batch, XtrainSOS)
            closest = np.argmin(dst, axis=1)
            ypred[batch_begin:batch_end] = ytrain[closest]

    error_rate = np.mean(ypred != ytest)
    print('Error Rate: %.2f%%' % (100*error_rate,))

    end_time = datetime.now()
    duration = end_time - start_time
    print('Total Time: %s' % duration)

    if use_flann:
        duration = end_time - start_time1
        print('Include FLANN build or load index: Total Time: %s' % duration)


if __name__ == "__main__":
    cwd = os.getcwd()
    if os.path.basename(cwd) == "demos":
        os.chdir(os.path.dirname(cwd))
    cwd = os.getcwd()
    mnist1NNDemo(test_size=1000, use_flann=False, perform_shuffle=True)