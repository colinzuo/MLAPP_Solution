from numpy.linalg import matrix_rank
import numpy as np

from toolbox.centerCols import *


def pcaPmtk(X, K=None, method=None):
    n, d = X.shape
    if method is None:
        cost = np.array([d**3, n**3, min(n*d**2, d*n**2)])
        method = np.argmin(cost) + 1
    methodNames = ['eig(Xt X)', 'eig(X Xt)', 'SVD(X)']
    print("Using method %s" % methodNames[method - 1])
    XCenter, mu = centerCols(X)
    if K is None:
        K = matrix_rank(XCenter)
    if method == 1:
        cov_matrix = np.cov(XCenter, rowvar=False, bias=True)
        evals, evec = np.linalg.eig(cov_matrix)
        sorted_idx = np.argsort(-evals)
        evals = evals[sorted_idx]
        evec = evec[:, sorted_idx]
        B = evec[:, 0:K]
    elif method == 2:
        w = np.dot(XCenter, XCenter.T)
        evals, evec = np.linalg.eig(w)
        sorted_idx = np.argsort(-evals)
        evals = evals[sorted_idx]
        evec = evec[:, sorted_idx]
        B = np.dot(np.dot(X.T, evec), np.diag(1. / np.sqrt(evals)))
        B = B[:, 0:K]
        evals = evals / n
        r = np.linalg.matrix_rank(XCenter)
        evals[r:] = 0
    elif method == 3:
        if n > d:
            full_matrices = False
        else:
            full_matrices = True
        u, s, vh = np.linalg.svd(XCenter, full_matrices=full_matrices)
        B = vh.T[:, 0:K]
        evals = 1/n * np.square(s)
    Z = np.dot(XCenter, B)
    Xrecon = np.dot(Z, B.T) + mu
    return B, Z, evals, Xrecon, mu


if __name__ == '__main__':
    # octave
    # X = [[2, 8, 3, 4, 9]; [6, 3, 5, 8, 2]; [1, 4, 5, 9, 3]; [5, 5, 1, 2, 7]]
    # [B, Z, evals, Xrecon, mu] = pcaPmtk(X, 2, 1)
    # -0.025365  - 0.78857
    # 0.35044    0.36752
    # -0.3313    0.23639
    # -0.60956    0.36123
    # 0.62866    0.23814
    X = np.array([[2, 8, 3, 4, 9], [6, 3, 5, 8, 2], [1, 4, 5, 9, 3], [5, 5, 1, 2, 7]])
    B, Z, evals, Xrecon, mu = pcaPmtk(X, 2, 3)
    print(B, Z)