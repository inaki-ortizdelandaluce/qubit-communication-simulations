import numpy as np
import scipy
from qt.qubit import Qubit


class PVM:
    def __init__(self, proj):
        """
        Creates a PVM with the specified rank-1 projectors.

        Parameters
        ---------
        proj : ndarray
            A 3-d array with the constituting rank-1 projectors.
       """
        # check input
        if not np.allclose(np.identity(2), np.sum(proj, axis=0)):
            raise ValueError('PVM projectors do not sum up the identity')

        self.proj = proj
        self.bloch = np.asarray([Qubit.density2bloch(p) for p in proj])

    @classmethod
    def new(cls, qubit: Qubit):
        """
        Creates a PVM with the rank-1 projectors corresponding to the specified qubit state.

        Parameters
        ---------
        qubit : Qubit
            The specified qubit state from which the two rank-1 projectors are generated.
        """
        rho = qubit.rho()
        sigma = np.identity(2) - rho
        proj = np.array([rho, sigma])
        return cls(proj)

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


class POVM:

    def __init__(self, weights, proj):
        """
        Creates a POVM with the specified weights and rank-1 projectors.

        Parameters
        ---------
        weights : ndarray
            The positive coefficients of the constituting rank-1 projectors.

        proj : ndarray
            A 3-d array with the constituting rank-1 projectors.
        """
        self.weights = weights
        self.elements = proj * weights[:, np.newaxis, np.newaxis]

        # check input
        if not np.allclose(np.identity(2), np.sum(self.elements, axis=0)):
            raise ValueError('POVM elements do not sum up the identity')

        positive = [np.all(np.linalg.eig(element)[0] >= -np.finfo(np.float32).eps) for element in self.elements]
        if not np.all(positive):
            raise ValueError('Some POVM elements are not definite positive')

        self.bloch = v = np.asarray([Qubit.density2bloch(p) for p in proj])

    @classmethod
    def new(cls, qubits):
        """
        Creates a POVM with the rank-1 projectors corresponding to the specified qubit states.

        Parameters
        ---------
        qubits : ndarray
           The specified array of N-2 qubit states from which the N rank-1 POVM projectors are generated.
        """
        # last element normalizes all POVM elements
        rhos = np.asarray([q.rho() for q in qubits])
        e = np.identity(2) - np.sum(rhos, axis=0)

        # diagonalize last element to obtain remaining rank-1 projectors
        _, w = np.linalg.eig(e)
        q1 = Qubit(w[:, 0])
        q2 = Qubit(w[:, 1])
        qubits = np.append(qubits, np.array([q1, q2]), axis=0)

        # compute POVM weights and elements as a linear program (see Sentís et al. 2013)
        v = np.asarray([q.bloch_vector() for q in qubits])
        n = len(qubits)

        a = np.vstack((np.ones((n,)), v.T))
        b = np.append(np.array([2]), np.zeros(3, ), axis=0)

        # c = np.zeros(n, ) finds a solution instead of minimizing a function
        # lower bounds set to 0.01, this could be fine-tuned
        lp = scipy.optimize.linprog(np.zeros(n, ), A_eq=a, b_eq=b, bounds=(0.01, 1), method='highs')
        _a, _e = lp['x'], np.asarray([q.rho() for q in qubits])

        return cls(_a, _e)

    def element(self, index):
        """
        Returns the POVM element for the corresponding index

        Parameters
        ---------
        index : the POVM element index

        Returns
        -------
        ndarray
            A 2-d array with the corresponding POVM element.
        """
        return self.element[index]

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
        rho = np.repeat(rho[np.newaxis, :, :], self.elements.shape[0], axis=0)

        # compute trace of projectors by density matrix
        return np.real(np.trace(np.matmul(self.elements, rho), axis1=1, axis2=2))

    def size(self):
        """
        Returns the number of POVM elements

        Returns
        -------
        int
            The number of POVM elements.
        """
        return np.size(self.elements, axis=0)

    def unitary(self):
        """
        Returns the associated unitary matrix in the extended Hilbert space according to Neumark's theorem

        Returns
        -------
        ndarray
            The nxn unitary matrix where n is the number of POVM elements.

        """
        d = 2
        n = self.size()
        u = np.zeros((n, n), dtype=np.complex_)

        # compute the kets of the rank-1 POVM projectors and assign to first d columns
        # v, _, _ = np.linalg.svd(self.elements, full_matrices=True, compute_uv=True, hermitian=False)
        # u[:, 0:d] = v[:, :, 0] / np.linalg.norm(v[:, :, 0], axis=0)
        w, v = np.linalg.eig(self.elements)

        v = v[np.where(w != 0)]  # FIXME
        u[:, 0:d] = v / np.linalg.norm(v, axis=0)

        # remaining n-d columns should correspond to orthogonal projectors in extended space
        p = np.eye(n, dtype=np.complex_)
        for idx in range(d):
            p -= np.outer(u[:, idx], u[:, idx].conj())

        counter = 0
        for b in np.eye(n, dtype=np.complex_):
            w = np.matmul(p, b)
            if not np.isclose(w, 0.0).all():
                w /= np.linalg.norm(w)
                u[:, counter + d] = w
                p -= np.outer(w, w.conj())
                counter += 1
            if counter == (n - d):
                break

        if not np.allclose(np.matmul(u, u.conj().T), np.eye(n)):
            raise ValueError('Neumark\'s square matrix is not unitary')

        return u
