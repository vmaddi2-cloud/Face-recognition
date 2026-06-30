"""
pca_eigenface.py
-----------------
Implements PCA-based face recognition exactly as specified in the
assignment (Turk & Pentland surrogate-covariance trick), using only
numpy/scipy for the linear algebra.

    Face_Db (mn x p)
        -> mean vector M (mn x 1)
        -> mean-zero data A (mn x p)
        -> surrogate covariance C = A^T A   (p x p)   instead of A A^T (mn x mn)
        -> eigen-decomposition of C  ->  eigenvectors V (p x p)
        -> pick best k eigenvectors  ->  feature vector W (p x k)
        -> eigenfaces U = W^T A^T   (k x mn)   (project mean-aligned faces
           back to image space, Turk & Pentland Eq.)
        -> signatures Omega = U * A  (k x p) -- one k-dim "signature" per face
"""
import numpy as np
from scipy.linalg import eigh


class EigenfaceModel:
    def __init__(self, k=None):
        self.k = k
        self.mean_ = None        # (mn, 1)
        self.eigenfaces_ = None  # (k, mn)
        self.signatures_ = None  # (k, p)
        self.eigvals_ = None     # (p,) full sorted eigenvalues (for the scree/k study)

    def fit(self, Face_Db, k=None):
        """
        Face_Db: (mn, p) ndarray, columns are face images (flattened).
        """
        if k is not None:
            self.k = k
        mn, p = Face_Db.shape

        # 2. Mean
        M = Face_Db.mean(axis=1, keepdims=True)  # (mn, 1)
        self.mean_ = M

        # 3. Mean-zero
        A = Face_Db - M  # (mn, p)

        # 4. Surrogate covariance (p x p) instead of (mn x mn)
        C = A.T @ A  # (p, p)

        # 5. Eigen decomposition (symmetric -> eigh, ascending order)
        eigvals, eigvecs = eigh(C)  # eigvals ascending, eigvecs (p, p)

        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # numerical cleanup: discard (near) zero / negative eigenvalues
        eps = 1e-8 * eigvals[0]
        valid = eigvals > eps
        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]
        self.eigvals_ = eigvals

        k_use = self.k if self.k is not None else len(eigvals)
        k_use = min(k_use, eigvecs.shape[1])
        self.k = k_use

        # 6. Feature vector: best k eigenvectors (p, k)
        W = eigvecs[:, :k_use]

        # 7. Generate eigenfaces: project mean-aligned faces to feature vector
        # U (k, mn) = W^T (k,p) @ A^T (p, mn)
        U = W.T @ A.T  # (k, mn)
        # normalize each eigenface to unit length (standard Eigenface practice)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U = U / norms
        self.eigenfaces_ = U

        # 8. Signatures: project every mean-aligned face onto the eigenfaces
        Omega = U @ A  # (k, p)
        self.signatures_ = Omega

        return self

    def project(self, I):
        """
        Project one or more test images onto the eigenface space.
        I: (mn,) or (mn, n_test) ndarray of RAW (unflattened-already) images.
        Returns projected signatures, shape (k,) or (k, n_test).
        """
        single = (I.ndim == 1)
        if single:
            I = I[:, None]
        I2 = I - self.mean_           # mean-zero
        proj = self.eigenfaces_ @ I2  # (k, n_test)
        return proj[:, 0] if single else proj

    def reconstruction_error(self, I):
        """
        Distance-from-face-space: how well the eigenfaces explain image I.
        Used to flag imposters (faces unlike anything seen in training).
        """
        single = (I.ndim == 1)
        if single:
            I = I[:, None]
        I2 = I - self.mean_
        proj = self.eigenfaces_ @ I2                  # (k, n)
        recon = self.eigenfaces_.T @ proj             # (mn, n) back to image space
        err = np.linalg.norm(I2 - recon, axis=0)
        return err[0] if single else err
