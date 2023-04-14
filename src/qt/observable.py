import numpy as np
import scipy as sp


class Observable:
    def __init__(self, matrix):

        if not sp.linalg.ishermitian(matrix):
            raise ValueError('Input matrix is not hermitian')

        self.matrix = matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eig(matrix)

    def eigen(self):
        return self.eigenvalues, self.eigenvectors

    def eigenvector(self, eigenvalue):
        m = self.eigenvectors[:, np.where(self.eigenvalues == eigenvalue)]
        m = m.reshape(m.size, )
        if m.size == 0:
            return None
        else:
            return m

