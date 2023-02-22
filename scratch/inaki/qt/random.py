import math
import numpy as np
from qt.qubit import Qubit
from qt.measurement import PVM


def vector3():
    """
    Generates a normalised vector uniformly distributed on the unit radius sphere

    Returns
    -------
    ndarray
            A normalised vector on the unit radius sphere

    """
    theta, phi = qubit().bloch_angles()
    return np.array([math.sin(theta) * math.cos(phi),
                     math.sin(theta) * math.sin(phi),
                     math.cos(theta)])


def qubit():
    """
    Generates a random qubit.

    Returns
    -------
    Qubit
        A random Qubit.
    """
    # evolve the zero state with a random unitary matrix
    # same as returning first column of random unitary matrix
    u = unitary((2, 2))
    return Qubit(u[:, 0])


def pvm():
    """
    Generates a random projection value measure for a qubit

    Returns
    -------
    PVM
            A projection value measure instance.
    """
    u = unitary((2, 2))

    # each column of the unitary random matrix is an orthogonal vector of a PVM basis
    measurement = PVM(u.T)
    return measurement


def pvm_vectors():
    """
    Generates a random projection value measure for a qubit in form of orthogonal bloch vectors

    Returns
    -------
    (ndarray, ndarray)
            A projection value measure consisting of tuple of projection operators in the form of bloch vectors.
    """
    # FIXME Use PVM class instead
    u = unitary((2, 2))
    return Qubit(u[:, 0]).bloch_vector(), \
        Qubit(u[:, 1]).bloch_vector()


def unitary(shape):
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
