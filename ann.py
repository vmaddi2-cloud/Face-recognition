"""
ann.py
------
A small feed-forward neural network (1 hidden layer) trained with plain
numpy back-propagation, used to classify the PCA "signatures" produced by
pca_eigenface.py. Only numpy is used for the math, per the assignment's
library restriction.
"""
import numpy as np


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SimpleANN:
    """
    Architecture: input -> hidden (tanh) -> output (softmax), trained with
    mini-batch gradient descent and cross-entropy loss.
    """

    def __init__(self, n_in, n_hidden, n_out, seed=42, lr=0.05):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        self.lr = lr
        self.n_out = n_out

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def predict_proba(self, X):
        _, _, _, a2 = self.forward(X)
        return a2

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X, y, epochs=400, batch_size=16, l2=1e-4, verbose=False):
        n = X.shape[0]
        Y = one_hot(y, self.n_out)
        rng = np.random.default_rng(0)
        loss_history = []

        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_shuf, Y_shuf = X[perm], Y[perm]

            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = Y_shuf[start:start + batch_size]
                m = xb.shape[0]

                z1, a1, z2, a2 = self.forward(xb)

                # cross-entropy gradient wrt z2 (softmax + CE simplifies nicely)
                dz2 = (a2 - yb) / m
                dW2 = a1.T @ dz2 + l2 * self.W2
                db2 = dz2.sum(axis=0, keepdims=True)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (1 - a1 ** 2)  # tanh derivative
                dW1 = xb.T @ dz1 + l2 * self.W1
                db1 = dz1.sum(axis=0, keepdims=True)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                _, _, _, a2_full = self.forward(X)
                eps = 1e-12
                ce = -np.mean(np.sum(Y * np.log(a2_full + eps), axis=1))
                loss_history.append(ce)
                print(f"    epoch {epoch:4d}  loss={ce:.4f}")

        return self
