import numpy as np


def centerCols(X, mu=None):
    if mu is None:
        mu = np.mean(X, axis=0)
    Y = X - mu
    return Y, mu