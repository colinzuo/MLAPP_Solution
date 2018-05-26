import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from toolbox.knn import *


def knnClassifyDemo():
    data_path = os.path.join("..", "data", "knnClassify3c", "knnClassify3c.mat")
    with h5py.File(data_path, 'r') as f:
        Xtrain = np.array(f['Xtrain']).T
        Xtest = np.array(f['Xtest']).T
        ytrain = np.array(f['ytrain'], dtype=int).T
        ytest = np.array(f['ytest'], dtype=int).T

    plt.ion()
    plt.show()

    plotLabeledData(Xtrain, ytrain)
    xval_range = [np.min(Xtrain[:, 0]), np.max(Xtrain[:, 0]), np.min(Xtrain[:, 1]), np.max(Xtrain[:, 1])]
    xgap = (xval_range[1] - xval_range[0]) / 20
    ygap = (xval_range[3] - xval_range[2]) / 20
    plt.xlim((xval_range[0]-xgap, xval_range[1]+xgap))
    plt.ylim((xval_range[2]-ygap, xval_range[3]+ygap))
    plt.title("knnClassifyTrainData")
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    plotLabeledData(Xtest, ytest)
    xval_range = [np.min(Xtest[:, 0]), np.max(Xtest[:, 0]), np.min(Xtest[:, 1]), np.max(Xtest[:, 1])]
    xgap = (xval_range[1] - xval_range[0]) / 20
    ygap = (xval_range[3] - xval_range[2]) / 20
    plt.xlim((xval_range[0] - xgap, xval_range[1] + xgap))
    plt.ylim((xval_range[2] - ygap, xval_range[3] + ygap))
    plt.title("knnClassifyTestData")
    plt.draw()
    plt.pause(0.001)

    Ks = [1, 5, 10, 20, 30, 40, 50, 100, 120]
    err_rate_tests = []
    err_rate_trains = []

    for K in Ks:
        model = KnnModel()
        model.fit(Xtrain, ytrain, K)
        yhat, _ = model.predict(Xtest)
        err_rate_test = np.mean(ytest != yhat)
        err_rate_tests.append(err_rate_test)

        yhat, _ = model.predict(Xtrain)
        err_rate_train = np.mean(ytrain != yhat)
        err_rate_trains.append(err_rate_train)

    plt.figure()
    plt.plot(Ks, err_rate_trains, 'bs:', Ks, err_rate_tests, 'rx-', linewidth=3, markersize=20)
    plt.legend(['train', 'test'])
    plt.xlabel('K')
    plt.ylabel('misclassification rate')
    plt.title('knnClassifyErrVsK')
    plt.draw()
    plt.pause(0.001)

    err_rate_tests = []
    cv_err_rate_trains = []
    cv_err_rate_train_ses = []
    perm = np.random.permutation(Xtrain.shape[0])
    Xtrain = np.ascontiguousarray(Xtrain[perm, :])
    ytrain = np.ascontiguousarray(ytrain[perm, :])
    nfolds = 5
    N = Xtrain.shape[0]
    fold_size = N//nfolds
    for K in Ks:
        model = KnnModel()
        model.fit(Xtrain, ytrain, K)
        yhat, _ = model.predict(Xtest)
        err_rate_test = np.mean(ytest != yhat)
        err_rate_tests.append(err_rate_test)

        err_rate_train_folds = []
        for fold in range(nfolds):
            if fold == 0:
                XtrainFold = Xtrain[fold_size:, :]
                ytrainFold = ytrain[fold_size:, :]
            elif fold == (nfolds-1):
                XtrainFold = Xtrain[:(nfolds-1)*fold_size, :]
                ytrainFold = ytrain[:(nfolds - 1) * fold_size, :]
            else:
                XtrainFold = np.vstack([Xtrain[0:fold * fold_size, :], Xtrain[(fold + 1) * fold_size:, :]])
                ytrainFold = np.vstack([ytrain[0:fold * fold_size, :], ytrain[(fold + 1) * fold_size:, :]])
            XtrainFold = np.ascontiguousarray(XtrainFold)
            ytrainFold = np.ascontiguousarray(ytrainFold)
            model.fit(XtrainFold, ytrainFold, K)
            yhat, _ = model.predict(Xtrain[fold*fold_size:(fold + 1) * fold_size, :])
            err_rate_train_fold = np.mean(ytrain[fold*fold_size:(fold + 1) * fold_size, :] != yhat)
            err_rate_train_folds.append(err_rate_train_fold)
        err_rate_train_folds = np.array(err_rate_train_folds)
        err_rate_train = np.mean(err_rate_train_folds)
        err_rate_train_se = np.std(err_rate_train_folds)
        cv_err_rate_trains.append(err_rate_train)
        cv_err_rate_train_ses.append(err_rate_train_se)

    plt.figure()
    plt.plot(Ks, cv_err_rate_trains, 'bs:', Ks, err_rate_tests, 'rx-', linewidth=3, markersize=20)
    plt.legend(['cv train', 'test'])
    plt.xlabel('K')
    plt.ylabel('misclassification rate')
    plt.title('knnClassifyErrVsK CrossValidation')
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    plt.errorbar(Ks, cv_err_rate_trains, yerr=cv_err_rate_train_ses, fmt='ko-')
    plt.xlabel('K')
    plt.ylabel('misclassification rate')
    plt.title('%d-fold cross validation, ntrain = %d' % (nfolds, N))
    min_idx = np.argmin(cv_err_rate_trains)
    plt.axvline(x=Ks[min_idx]+2)
    plt.draw()
    plt.pause(0.001)

    plt.ioff()
    plt.show()


def plotLabeledData(X, y):
    colors = ['r', 'b', 'g']
    markers = ['+', '*', 'x']
    C = np.max(y)
    for c in range(1, C+1):
        ndx = np.where(y[:,0] == c)
        plt.scatter(X[ndx, 0], X[ndx, 1], s=12, c=colors[c-1], marker=markers[c-1])


if __name__ == "__main__":
    cwd = os.getcwd()
    if os.path.basename(cwd) == "demos":
        os.chdir(os.path.dirname(cwd))
    cwd = os.getcwd()
    knnClassifyDemo()
    input("Press [enter] to continue.")