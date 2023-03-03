import numpy as np
from qt.qubit import Qubit


class PVM:
    def __init__(self, qubit: Qubit):
        """
        Initializes a PVM with the elements corresponding to the specified qubit state.

        Parameters
        ---------
        qubit : Qubit
            The specified qubit state from which the rank-1 projectors are generated.
        """
        rho = qubit.rho()
        sigma = np.identity(2) - rho
        self.bloch = np.array([Qubit.density2bloch(rho), Qubit.density2bloch(sigma)])
        self.proj = np.array([rho, sigma])

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

    def probability(self, qubit: Qubit):
        """
        Returns the probabilities of the different outcomes for a given qubit state

        Parameters
        ---------
        qubit : the qubit state

        Returns
        -------
        ndarray
                The probabilities for each outcome given the input state, stored in a 1-d array.
        """

        # repeat density matrix along zero axis
        rho = qubit.rho()
        rho = np.repeat(rho[np.newaxis, :, :], self.proj.shape[0], axis=0)

        # compute trace of projectors by density matrix
        return np.real(np.trace(np.matmul(self.proj, rho), axis1=1, axis2=2))
