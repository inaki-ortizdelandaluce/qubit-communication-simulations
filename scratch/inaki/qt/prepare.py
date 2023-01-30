import numpy as np
from qt.qubit import Qubit


def random_qubit():
    """
    Generates a random qubit.

    Returns
    -------
    Qubit
        A random Qubit.
    """
    # evolve the zero state with a random unitary matrix
    # same as returning first column of random unitary matrix
    unitary = random_unitary((2, 2))
    return Qubit.from_array(unitary[:, 0])


def random_qubit_pvm():
    """
    Generates a random projection value measure for a qubit

    Returns
    -------
    (ndarray, ndarray)
            A projection value measure consisting of tuple of projection operators for the first and second
            random basis elements.
    """
    unitary = random_unitary((2, 2))
    return Qubit.from_array(unitary[:, 0]).to_density_matrix(), \
        Qubit.from_array(unitary[:, 1]).to_density_matrix()


def random_unitary(shape):
    """
    Generates a random unitary matrix with the given shape.

    Parameters
    ---------
    shape : int or tuple of ints
        Shape of the unitary matrix.

    Returns
    -------
    ndarray
        Unitary matrix with the given shape.
    """
    # build random complex matrix
    m = np.random.normal(0, 1, shape) + 1.j * np.random.normal(0, 1, shape)

    # apply Gram-Schmidt QR decomposition to orthogonalize the matrix
    q, *_ = np.linalg.qr(m, mode='complete')

    return q
