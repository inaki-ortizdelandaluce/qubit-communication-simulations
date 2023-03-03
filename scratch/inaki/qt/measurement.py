import numpy as np


class PVM:
    def __init__(self, basis=np.array([[1, 0], [0, 1]])):
        """
        Initializes a PVM with the rank-1 projector of the specified basis.
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
        self.proj = np.zeros((self.basis.shape[0], *self.basis.shape), dtype=np.complex_)
        for i in range(self.basis.shape[0]):
            self.proj[i] = np.outer(self.basis[i], self.basis[i].conj())

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

    def probability(self, rho):
        """
        Returns the probabilities of the different outcomes for a given state

        Parameters
        ---------
        rho : the state in density matrix form

        Returns
        -------
        ndarray
                The probabilities for each outcome given the input state, stored in a 1-d array.
        """

        # repeat density matrix along zero axis
        rho = np.repeat(rho[np.newaxis, :, :], self.proj.shape[0], axis=0)

        # compute trace of projectors by density matrix
        return np.real(np.trace(np.matmul(self.proj, rho), axis1=1, axis2=2))
