import numpy as np
from qt.qubit import Qubit


class PVM:
    def __init__(self, basis=np.array([[1, 0], [0, 1]])):
        """
        Initializes a PVM with the rank-1 projectors of the specified vector basis.
        If no argument is provided, it creates a PVM with the rank-1 projectors of the computational basis in a
        two-dimensional space.

        Parameters
        ---------
        basis : ndarray
            The specified vector basis from which the rank-1 projectors are generated. The basis is a (n,n) shaped
            2-d array with n the dimension of the basis, where each row is an element of the basis.
        """

        # normalise basis
        self.basis = np.divide(basis.T, np.linalg.norm(basis, axis=1)).T

        # build projectors
        self.proj = np.zeros((basis.shape[0], *basis.shape), dtype=np.complex_)
        for i in range(basis.shape[0]):
            self.proj[i] = np.outer(basis[i], basis[i].conj())

    def projector(self, index):
        """
        Returns rank-1 projector for the corresponding index

        Parameters
        ---------
        index : the projector index

        Returns
        -------
        ndarray
                A 2-d array with the corresponding projector.
        """
        return self.proj[index]

