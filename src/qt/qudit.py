import numpy as np
from qt.qubit import Qubit


class Qudit:
    def __init__(self, ket):
        """
        Initializes a qudit in the computational basis.

        Parameters
        ---------
        ket : ndarray
            The qudit components in the computational basis in a 1-d complex array.
        """
        self.ket = ket
        self.normalize()

    def normalize(self):
        self.ket = self.ket/np.linalg.norm(self.ket)

    def rho(self):
        """
        Returns the density matrix corresponding to the qudit in a pure state.

        Returns
        -------
        ndarray
            A nxn density matrix corresponding to the qubit in a pure state.
        """
        return np.outer(self.ket, self.ket.conj())

    @classmethod
    def bipartite(cls, q1: Qubit, q2: Qubit):
        ket = np.tensordot(q1.ket(), q2.ket(), axes=0).reshape(4, )
        return cls(ket)

