import numpy as np


def oneOfK(y, K=None):
    unique, ymapped = np.unique(y, return_inverse=True)
    nunique = np.size(unique)
    if K is not None and nunique != K:
        if K < nunique:
            print("K is less than the number of unique labels in y - oneOfK does not know which ones to remove.")
        ymapped = ymapped + unique[0] - 1
        unique = np.arange(1, K+1)
    else:
        K = nunique
    N = ymapped.shape[0]
    yy = np.zeros((N, K))
    yy[np.arange(N), ymapped] = 1
    return yy, unique


if __name__ == "__main__":
    yy, map = oneOfK([1, 2, 1, 3], 4)
    print(yy)
    print(map)
    print("")

    yy, map = oneOfK(['yes', 'no', 'yes', 'maybe'])
    print(yy)
    print(map)
    print("")

    yy, map = oneOfK([-1,-1,1,-1,-1,1,-1])
    print(yy)
    print(map)
    print("")
