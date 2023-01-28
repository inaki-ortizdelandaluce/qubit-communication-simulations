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
    unitary = random_unitary((2, 2))
    q = np.matmul(unitary, Qubit().to_array())
    return Qubit.from_array(q)


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
