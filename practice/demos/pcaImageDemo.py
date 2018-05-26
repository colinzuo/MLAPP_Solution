import scipy.io
import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from toolbox.pcaPmtk import *

data_path = os.path.join("..", "bigData", "olivettiFaces", "olivettiFaces.mat")
mat = scipy.io.loadmat(data_path)

X = mat['faces'].T
y = np.matlib.repmat(np.arange(1, 41), 10, 1).flatten(order='F')

n, d = X.shape
h, w = 64, 64
name = 'faces'

perm = np.random.permutation(n)

f, ax_arr = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax_arr[i, j].imshow(X[perm[i*2+j], :].reshape(h, w).T, cmap="gray")
        ax_arr[i, j].axis('off')
f.suptitle("pcaImages-%s-images" % name)
plt.show()

XC = X - np.mean(X, axis=0)
print('Performing PCA.... stay tuned')

V, Z, evals, _, mu = pcaPmtk(X)

f, ax_arr = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            ax_arr[i, j].imshow(mu.reshape(h, w).T, cmap="gray")
            ax_arr[i, j].axis('off')
            ax_arr[i, j].set_title('mean')
        else:
            idx = i*2+j-1
            ax_arr[i, j].imshow(V[:, idx].reshape(h, w).T, cmap="gray")
            ax_arr[i, j].axis('off')
            ax_arr[i, j].set_title('principal basis %d' % idx)
f.suptitle("pcaImages-%s-basis" % name)
plt.show()

# Plot reconstructed image
ndx = 125
Ks = [5, 10, 20, np.linalg.matrix_rank(XC)]

f, ax_arr = plt.subplots(2, 2)
for ki in range(len(Ks)):
    k = Ks[ki]
    Xrecon = np.dot(Z[ndx, 0:k], V[:, 0:k].T) + mu
    i = ki//2
    j = ki%2
    ax_arr[i, j].imshow(Xrecon.reshape(h, w).T, cmap="gray")
    ax_arr[i, j].axis('off')
    ax_arr[i, j].set_title('Using %d bases' % k)
f.suptitle("pcaImages-%s-reconImages" % name)
plt.show()

Ks = []
Ks += range(1, 10)
Ks += range(10, 50, 5)
Ks += range(50, np.linalg.matrix_rank(XC), 25)
mse = []

for ki in range(len(Ks)):
    k = Ks[ki]
    Xrecon = np.dot(Z[:, 0:k], V[:, 0:k].T) + mu
    err = Xrecon - X
    mse.append(np.sqrt(np.mean(np.power(err, 2))))
plt.plot(Ks, mse, '-o')
plt.ylabel("mse")
plt.xlabel("K")
plt.title('reconstruction error')
plt.show()

plt.plot(np.cumsum(evals)/np.sum(evals), 'ko-')
plt.show()
